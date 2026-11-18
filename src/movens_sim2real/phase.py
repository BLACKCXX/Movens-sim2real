"""Causal four-stage labels and independent continuous release acceptance."""

from dataclasses import dataclass, field

import numpy as np


RULE_VERSION = "apple-causal-v3.1"
REASONS = ("steady", "near_closing", "finger_contact", "stable_lift",
           "above_goal", "target_supported", "lost_grip", "contact_lost",
           "retry", "unknown")
NEAR, GRASP, CARRY, PLACE = 1, 2, 3, 4


@dataclass
class Phase:
    rules: dict
    stage: int = NEAR
    stage_age: float = 0.
    t: float | None = None
    initial_z: float | None = None
    timers: dict = field(default_factory=dict)
    lift_xy: np.ndarray | None = None
    grip_rel: np.ndarray | None = None
    ever_grasped: bool = False
    ever_carried: bool = False
    carried_now: bool = False
    goal_now: bool = False
    near_reached: bool = False
    slip_count: int = 0
    max_stable_s: float = 0.
    still_s: float = 0.
    last_event: str = "steady"

    def timer(self, name, ok, dt):
        # True at two observations is required to count the intervening time.
        # None is an evidence gap and cannot extend a continuity window.
        previous, elapsed = self.timers.get(name, (False, 0.))
        elapsed = elapsed+dt if ok and previous else 0.
        self.timers[name] = (ok is True or isinstance(ok, np.bool_) and bool(ok), elapsed)
        return elapsed

    def advance(self, t, obs, closing, *, stable_interval=None):
        """One real decision boundary. Stable evidence covers the *past* interval.

        Optional cues are represented by explicit masks. A missing interval
        cannot advance a timer or qualify a release, even if its endpoints look good.
        """
        dt = 0. if self.t is None else t - self.t
        if not np.isfinite(t) or (self.t is not None and (dt <= 0 or dt > .10000001)):
            raise ValueError(f"Phase time must be continuous within a reset: {self.t} -> {t}")
        self.t = t
        self.stage_age += dt
        obj = obs["object"]
        object_valid = bool(obs.get("object_valid", True))
        rel_valid = object_valid and bool(obs.get("wrist_valid", True))
        support_valid = bool(obs.get("support_valid", True))
        groups = obs["groups"]
        known = obs["groups_valid"]
        finger = bool(np.any(groups & known))
        hand = bool(obs["robot_contact"])
        hand_valid = bool(obs.get("robot_contact_valid", True))
        surface = bool(obs["original_support"])
        supported = bool(obs["target_support"])
        if self.initial_z is None and object_valid:
            self.initial_z = float(obj[2])
        rel = obj[:3]-obs["wrist"][:3] if rel_valid else None
        rel_speed = np.linalg.norm(obs["twist"][:3]-obs["wrist_twist"][:3]) if rel_valid else np.inf
        diameter = obs["diameter"]
        tips_valid = bool(obs.get("tips_valid", True))
        close = bool(tips_valid and rel_valid and obs["tips"][:, 0].min() <= self.rules["near_diameters"]*diameter)
        self.near_reached |= close
        close_time = self.timer("near", close and closing, dt)
        out = "steady"
        slip = False
        if self.stage == NEAR:
            if finger or close_time+1e-9 >= self.rules["near_seconds"]:
                self.stage = GRASP
                out = "finger_contact" if finger else "near_closing"
        elif self.stage == GRASP:
            lifted = (object_valid and self.initial_z is not None and
                      obj[2]-self.initial_z >= max(self.rules["lift_m"], self.rules["lift_diameters"]*diameter))
            group_grasp = bool(known[2] and groups[2] and np.any(groups[:2] & known[:2]))
            held = bool(lifted and support_valid and not surface and not supported and group_grasp
                        and rel_speed < self.rules["hold_rel_speed_m_s"])
            if self.timer("hold", held, dt)+1e-9 >= self.rules["hold_seconds"]:
                self.stage = CARRY
                self.ever_grasped = True
                self.lift_xy = obj[:2].copy()
                self.grip_rel = rel.copy()
                self.goal_now = False
                out = "stable_lift"
            elif self.timer("miss", tips_valid and not close and hand_valid and not hand, dt) >= .2:
                self.stage = NEAR
                out = "retry"
        else:
            if rel is not None and self.lift_xy is not None:
                self.carried_now |= np.linalg.norm(obj[:2]-self.lift_xy) >= self.rules["transport_m"]
                self.ever_carried |= self.carried_now
            near_goal = bool(object_valid and obs.get("margin_valid", True)
                             and obs["margin"] >= -self.rules["goal_extra_margin_m"])
            if self.stage == CARRY:
                goal = (support_valid and supported) or (near_goal and obs["twist"][2] < -.005)
                if self.carried_now and self.timer("goal", goal, dt)+1e-9 >= self.rules["goal_seconds"]:
                    self.stage = PLACE
                    self.goal_now = True
                    out = "target_supported" if supported else "above_goal"
            drift = (rel is not None and self.grip_rel is not None and
                     np.linalg.norm(rel-self.grip_rel) >= self.rules["slip_m"])
            dropped = hand_valid and not hand and drift and (self.stage == CARRY or not near_goal)
            if self.timer("slip", dropped, dt)+1e-9 >= self.rules["slip_seconds"]:
                self.slip_count += 1
                self.stage = GRASP if close else NEAR
                self.lift_xy = self.grip_rel = None
                self.carried_now = self.goal_now = False
                slip = True
                out = "lost_grip"
        # A release witness is independent of the phase id; it qualifies only
        # after a verified grasp and transport in the same attempt.
        self.still_s = self.still_s+dt if self.carried_now and self.goal_now and stable_interval is True else 0.
        self.max_stable_s = max(self.max_stable_s, self.still_s)
        if out != "steady":
            self.stage_age = 0.
            self.timers.clear()
        valid = bool(rel_valid and support_valid and hand_valid and
                     (np.all(known) if self.stage == GRASP else tips_valid))
        if not valid and out == "steady":
            out = "unknown"
        self.last_event = out
        confidence = min(float(obs.get("confidence", 1.)), 1.) if valid else 0.
        return (self.stage, valid, confidence, REASONS.index(out), self.stage_age, self.still_s, slip)

    def outcome(self, complete, continuous=True, engineering_error=None):
        success = bool(complete and continuous and not engineering_error and self.ever_grasped and
                       self.carried_now and self.goal_now and self.still_s+1e-9 >= self.rules["stable_seconds"])
        if success:
            failure = None
        elif engineering_error:
            failure = "engineering_invalid"
        elif not complete or not continuous:
            failure = "incomplete_or_local_reset"
        elif not self.near_reached:
            failure = "wrist_outside_reach"
        elif not self.ever_grasped:
            failure = "no_stable_grasp"
        elif self.slip_count and not self.goal_now:
            failure = "slip_or_off_target_release" if self.ever_carried else "slip_before_transport"
        elif not self.ever_carried:
            failure = "no_transport"
        else:
            failure = "unstable_or_off_target_release"
        return dict(success=success, failure=failure, stable_grasp=self.ever_grasped, transported=self.ever_carried,
                    reached_goal=self.goal_now, slip_count=self.slip_count, reached_near=self.near_reached,
                    stable_release_s=self.still_s, max_stable_release_s=self.max_stable_s, rule_version=RULE_VERSION)


def phase_arrays(rows):
    ids, valid, confidence, reason, age, stable, slip = map(np.asarray, zip(*rows))
    return dict(phase_id=ids.astype(np.int8), phase_valid=valid.astype(bool),
                phase_confidence=confidence.astype(np.float32), transition_reason=reason.astype(np.int8),
                phase_age_s=age.astype(np.float32), stable_release_s=stable.astype(np.float32), slip=slip.astype(bool))

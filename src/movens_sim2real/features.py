"""Same typed value/validity/confidence inputs for sim, replay and sensors."""
import numpy as np

from .contract import LEFT, pose_rotation6, rotation, state_vector, wxyz

ACTOR_SIZE = 240
DYN_SIZE = 145    # 101 state, 43 previous executed targets, age
PHYS_SIZE = 29
PRED_SIZE = 98
PLAN_OFFSETS = (0, 5, 10)


def packed(value, valid=True, confidence=1.):
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    mask = np.broadcast_to(np.asarray(valid, dtype=bool), v.shape)
    cf = np.broadcast_to(np.asarray(confidence, dtype=np.float32), v.shape)
    if not np.isfinite(v[mask]).all() or not np.isfinite(cf).all() or (cf < 0).any() or (cf > 1).any():
        raise ValueError("Valid features must be finite, confidence in [0, 1]")
    return np.asarray([np.where(mask, v, 0.), mask, np.where(mask, cf, 0.)], dtype=np.float32)


def actor_frame(obs, targets, planned_wrist, prev_target, prev_correction, phase,
                task_left_s, chunk_age_s, *, ablation="phase"):
    """The only future information is an already-generated nominal block."""
    pieces = []
    def add(v, k=None, default=True, confidence=None):
        valid = obs.get(k+"_valid", default) if k else default
        c = obs.get(k+"_confidence", 1.) if k and confidence is None else 1. if confidence is None else confidence
        pieces.append(packed(v, valid, c))

    add(obs["q"], "q")
    add(obs["qd"], "qd")
    rel = np.zeros(15)
    obj_valid = bool(obs.get("object_valid", True))
    wrist_valid = bool(obs.get("wrist_valid", True))
    valid = wrist_valid and obj_valid
    if valid:
        r = rotation(obs["wrist"][3:]).inv()
        xyz = r.apply(obs["object"][:3]-obs["wrist"][:3])
        quat = wxyz(r*rotation(obs["object"][3:]))
        rel = np.r_[xyz, pose_rotation6(np.r_[xyz, quat]),
                    r.apply(obs["twist"][:3]-obs["wrist_twist"][:3]),
                    r.apply(obs["twist"][3:]-obs["wrist_twist"][3:])]
    add(rel, default=valid, confidence=min(obs.get("object_confidence", 1.), obs.get("wrist_confidence", 1.)))
    add(np.zeros(7) if prev_correction is None else prev_correction, default=prev_correction is not None)
    add(np.zeros(43) if prev_target is None else prev_target, default=prev_target is not None)
    add(obs["dims"], "dims")
    valid_target = bool(obs.get("target_valid", True) and obj_valid)
    rel_target = obs["target"][:3]-obs["object"][:3] if valid_target else np.zeros(3)
    # Height relative to the target support plane; estimated from target pose
    # at deployment with the corresponding validity and confidence.
    add(np.r_[rel_target, rel_target[2]], default=valid_target,
        confidence=min(obs.get("target_confidence", 1.), obs.get("object_confidence", 1.)))
    add(obs["tips"].ravel(), "tips")
    p, pvalid, pconf, _, age, _, _ = phase
    phase_onehot = np.eye(4)[p-1] if ablation == "phase" and p else np.zeros(4)
    if ablation == "time":
        phase_onehot = np.eye(4)[min(3, int(4*(1-np.clip(task_left_s/max(task_left_s+obs.get("time", 0.), .02), 0, 1))))]
    add(phase_onehot, default=(pvalid and bool(p)) if ablation == "phase" else ablation == "time",
        confidence=pconf if ablation == "phase" else 1.)
    add(np.r_[1., np.zeros(7), 1., 0.])  # single task index and left handedness
    add([age, obs.get("hold_elapsed_s", 0.)], default=bool(pvalid and ablation == "phase"), confidence=pconf)
    add([task_left_s, chunk_age_s, obs.get("age_s", 0.)])
    for j in PLAN_OFFSETS:
        present = j < len(targets)
        fw = np.r_[targets[j, LEFT], planned_wrist[j, :3], pose_rotation6(planned_wrist[j])] if present else np.zeros(16)
        add(fw, default=present)
        add([float(present)])
    result = np.concatenate(pieces, axis=1)
    if result.shape != (3, ACTOR_SIZE):
        raise ValueError(f"Actor feature schema mismatch {result.shape}")
    return result


def dyn_frame(obs, previous_target):
    x = state_vector(obs["q"], obs["qd"], obs["object"], obs["twist"])
    mask = np.r_[np.broadcast_to(obs.get("q_valid", True), (43,)), np.broadcast_to(obs.get("qd_valid", True), (43,)),
                 np.repeat(obs.get("object_valid", True), 15)]
    conf = np.r_[np.broadcast_to(obs.get("q_confidence", 1.), (43,)),
                 np.broadcast_to(obs.get("qd_confidence", 1.), (43,)), np.repeat(obs.get("object_confidence", 1.), 15)]
    u = np.zeros(43) if previous_target is None else previous_target
    return packed(np.r_[x, u, obs.get("age_s", 0.)], np.r_[mask, np.repeat(previous_target is not None, 43), True],
                  np.r_[conf, np.ones(44)])


def window(x, times, i, length, max_seconds=.48):
    start = max(i+1-length, np.searchsorted(times, times[i]-max_seconds-1e-8))
    n = i+1-start
    result = np.zeros((length, *x.shape[1:]), dtype=np.float32)
    result[:n] = x[start:i+1]
    return result, np.int64(n)


def point_context(points, valid=None, confidence=None):
    n = len(points)
    return dict(points=np.asarray(points, dtype=np.float32),
        point_valid=np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        point_conf=np.ones(n, dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32))

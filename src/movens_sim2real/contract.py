"""Time, coordinate and action contracts shared by capture, learning and replay."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "apple-43-v1"
LEFT = slice(29, 36)
GROUPS = ("index", "middle", "thumb")
EVENTS = (*GROUPS, "slip", "plate_support")


def read(path):
    return json.loads(Path(path).read_text())


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, default=_default, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def config(path=ROOT / "configs/experiment.json"):
    path = Path(path).resolve()
    cfg = read(path)
    for field in ("interact", "teacher", "replay", "sonic", "sonic_xml", "sonic_alternate_xml", "checkpoint", "baseline_eval"):
        cfg[field] = (path.parent.parent / cfg[field]).resolve()
    return cfg


def candidate(cfg, episode):
    return cfg["replay"] / f"episode_{int(episode):06d}" / "offset_000mm"


def split_map(groups):
    result = {}
    for split, sources in groups.items():
        for source in sources:
            if source in result:
                raise ValueError(f"Duplicate/unseparated source group {source}")
            result[source] = split
    return result


def extend_split(prior, ids, seed):
    """Keep published groups and split previously unseen groups before sampling."""
    result = dict(prior)
    ids = sorted(set(ids) - result.keys())
    order = np.random.default_rng(seed).permutation(len(ids))
    n_eval = len(ids) // 10
    for k, j in enumerate(order):
        result[ids[j]] = "val" if k < n_eval else "test" if k < 2 * n_eval else "train"
    return result


def exact_indices(saved, requested, source, atol=1e-8):
    saved, requested = np.asarray(saved), np.asarray(requested)
    if saved.ndim != 1 or len(saved) < 1 or not np.isfinite(saved).all() or np.any(np.diff(saved) <= 0):
        raise ValueError(f"{source}: state times must be finite and increasing")
    k = np.searchsorted(saved, requested - atol)
    if (not np.isfinite(requested).all() or np.any(k >= len(saved))
            or not np.allclose(saved[k], requested, atol=atol, rtol=0)):
        raise ValueError(f"{source}: missing actual samples at requested times")
    return k


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[..., [1, 2, 3, 0]])


def wxyz(rot):
    return rot.as_quat()[..., [3, 0, 1, 2]]


def pose_rotation6(pose):
    # First two columns, each contiguous.
    return rotation(pose[..., 3:]).as_matrix()[..., :, :2].swapaxes(-1, -2).reshape((*pose.shape[:-1], 6))


def state_vector(q, qd, pose, twist):
    return np.concatenate((q, qd, pose[..., :3], pose_rotation6(pose), twist), axis=-1)


def state_delta(q, qd, pose, twist, q1, qd1, pose1, twist1):
    d_rot = (rotation(pose[..., 3:]).inv() * rotation(pose1[..., 3:])).as_rotvec()
    return np.concatenate((q1-q, qd1-qd, pose1[..., :3]-pose[..., :3], d_rot, twist1-twist), axis=-1)


@dataclass(frozen=True)
class ActionLimits:
    lo: np.ndarray
    hi: np.ndarray
    scale: np.ndarray
    max_step: np.ndarray

    @classmethod
    def from_scene(cls, scene, fraction, dt):
        lo, hi = scene.sim.joint_limits.T.copy()
        # Native target bounds are intersected with actuator bounds. The
        # existing control clips Body/other hand without adding a slew filter.
        model = scene.sim.model
        ids = scene.sim.actuator_ids
        limited = model.actuator_ctrllimited[ids].astype(bool)
        lo[limited] = np.maximum(lo[limited], model.actuator_ctrlrange[ids[limited], 0])
        hi[limited] = np.minimum(hi[limited], model.actuator_ctrlrange[ids[limited], 1])
        if np.any(lo >= hi):
            raise ValueError("Invalid joint/actuator intersection")
        scale = (hi[LEFT]-lo[LEFT])*fraction
        return cls(lo, hi, scale, scale * (dt/.02) * .5)

    def execute(self, nominal, residual, previous_correction):
        if nominal.shape != (43,) or residual.shape != (7,) or previous_correction.shape != (7,):
            raise ValueError("Expected 43 decoded targets and left-hand residual/correction of width 7")
        if not all(np.isfinite(x).all() for x in (nominal, residual, previous_correction)) or np.any(abs(residual) > 1.000001):
            raise ValueError("Nonfinite target or normalized residual outside [-1, 1]")
        base = np.clip(nominal, self.lo, self.hi)
        # Constrain slew of the correction, not of the preexisting Movens
        # interface. Exact zero is the original control if previous is zero.
        requested = np.clip(self.scale*residual, previous_correction-self.max_step,
                            previous_correction+self.max_step)
        desired = base.copy()
        desired[LEFT] = np.clip(base[LEFT]+requested, self.lo[LEFT], self.hi[LEFT])
        return desired, desired[LEFT]-base[LEFT]


def split_residual(executed, limits, rng, mode="mixed"):
    """Time-correlated plans; rederive labels after position and slew clipping."""
    action = np.asarray(executed)
    if action.ndim != 2 or action.shape[1] != 7 or not np.isfinite(action).all():
        raise ValueError("Expected contiguous finite executed left-hand targets")
    lo, hi = limits.lo[LEFT], limits.hi[LEFT]
    if (action < lo-1e-7).any() or (action > hi+1e-7).any():
        raise ValueError("Demonstrated target exceeds the native bounds")
    if mode == "mixed":
        mode = rng.choice(("correct", "perturbed", "zero"), p=(.15, .80, .05))
    legal_zero = (np.all(lo <= 0) and np.all(hi >= 0) and
                  np.all(abs(action) <= limits.scale) and
                  np.all(abs(np.diff(np.vstack((np.zeros((1, 7)), action)), axis=0)) <= limits.max_step))
    if mode == "zero" and not legal_zero:
        mode = "perturbed"
    if mode == "correct":
        nominal = action.copy()
    elif mode == "zero":
        nominal = np.zeros_like(action)
    elif mode == "perturbed":
        # Tighten the per-tick feasible intervals backward so the sampled
        # residual can always reach the next interval under the slew limit.
        low = np.maximum(-limits.scale, action-hi)
        high = np.minimum(limits.scale, action-lo)
        for i in range(len(action)-2, -1, -1):
            low[i] = np.maximum(low[i], low[i+1]-limits.max_step)
            high[i] = np.minimum(high[i], high[i+1]+limits.max_step)
        error = np.zeros(7)
        nominal = np.empty_like(action)
        for i, target in enumerate(action):
            error = np.clip(.9*error + rng.normal(0, .35, 7)*limits.scale,
                            np.maximum(low[i], error-limits.max_step),
                            np.minimum(high[i], error+limits.max_step))
            nominal[i] = target-error
    elif mode == "zero_ablation":
        nominal = np.zeros_like(action)
    else:
        raise ValueError(f"Unknown residual dataset mode: {mode}")
    label = (action - nominal) / limits.scale
    return nominal, label, mode

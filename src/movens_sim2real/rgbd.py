"""Registered RGB-D capture and time-aware delivery to the observation adapter."""
from pathlib import Path

import numpy as np

from .contract import read, sha256
from .data import array_dict
from .hardware import DepthTrack, Latest


def recorded_depth(path, rgb_source, rgb_s, shape, *, domain):
    meta = read(path)
    if (meta.get("format") != "fixed-base-registered-rgbd-v1" or meta.get("domain") != domain
            or meta.get("depth_unit") != "m" or meta.get("registration") != "rectified_ego_rgb"
            or not meta.get("clock_source") or not meta.get("calibration_source")
            or meta.get("rgb_sha256") != sha256(rgb_source)):
        raise ValueError(f"{path}: expected registered metric depth bound to this RGB and capture clock")
    source = Path(path).parent / meta["arrays"]
    if sha256(source) != meta["arrays_sha256"]:
        raise ValueError(f"{source}: registered depth archive changed")
    a = array_dict(source)
    idx, ts, ready, depth = (a[k] for k in ("rgb_index", "captured_s", "available_s", "depth_m"))
    if (idx.ndim != 1 or not len(idx) or not np.issubdtype(idx.dtype, np.integer)
            or idx[0] < 0 or idx[-1] >= len(rgb_s) or np.any(np.diff(idx) <= 0)
            or ts.shape != idx.shape or ready.shape != idx.shape or not np.isfinite(np.r_[ts,ready]).all()
            or np.any(np.diff(ts) <= 0) or np.any(np.diff(ready) <= 0)
            or np.any(abs(ts - rgb_s[idx]) > .012) or np.any(ready < np.maximum(ts, rgb_s[idx]))
            or depth.shape != (len(ts), *shape) or not np.issubdtype(depth.dtype, np.floating)):
        raise ValueError(f"{source}: invalid RGB/depth pair, availability clock or metric pixel grid")
    return meta, a


class DepthObservations:
    """Publish depth estimates outside the control loop; consume by causal availability and capture age."""
    def __init__(self, intrinsics, base_from_optical, diameter_m):
        self.track = DepthTrack(intrinsics, base_from_optical, diameter_m)
        self.samples = Latest(.15)

    def publish(self, marker, depth_m, depth_s, available_s, pose=None):
        rgb_s = marker["captured_s"]
        capture_s = (rgb_s, depth_s) if pose is None else (rgb_s, depth_s, pose["captured_s"])
        if not np.isfinite((*capture_s, available_s)).all() or available_s < max(capture_s):
            raise ValueError("A perception result cannot be available before any of its measurements")
        p = self.track(depth_m, marker, depth_s, pose_estimate=pose)
        if not self.samples.put(available_s, (p, min(capture_s))):
            raise ValueError("Out-of-order perception delivery")

    def read(self, now):
        item = self.samples.read(now)
        if item is not None:
            value, captured = item[0]
            age = now - captured
            if value is not None and age <= self.samples.max_age:
                return value, captured, age

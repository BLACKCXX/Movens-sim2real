"""Verify the indexed Apple and SONIC data against the original executions."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from movens_sim2real.contract import LEFT, ActionLimits, config, read, split_map, split_residual, write
from movens_sim2real.data import array_dict, labels, obs_at, raw_targets
from movens_sim2real.phase import Phase


def equal(actual, expected, path, field):
    if not np.array_equal(actual, expected):
        raise ValueError(f"{path}: {field} differs")


def validate_apple(row, manifest, cfg):
    identity = row["identity"]
    parent = manifest[identity["episode"]]
    path = Path(row["path"])
    if any(identity[k] != parent[k] for k in ("source_group_id", "split", "reset_id")):
        raise ValueError(f"{path}: source split changed")
    a = array_dict(path)
    source = Path(parent["source"])
    names, clock, original = raw_targets(source, cfg)
    t, ticks = a["t"], a["control_ticks"]
    dt = cfg["control_dt"]
    equal(a["joint_names"].tolist(), names, path, "joint names")
    equal(np.sort(a["actuator_indices"]), np.arange(43), path, "actuator permutation")
    if not np.allclose(t, ticks * dt, rtol=0, atol=1e-8) or np.any(np.diff(ticks) <= 0):
        raise ValueError(f"{path}: incorrect state clock")
    equal(a["command_start_s"], t[:-1], path, "command start")
    equal(a["command_end_s"], t[1:], path, "command end")
    equal(a["actual_20ms"], np.diff(ticks) == 1, path, "one-control-step transition mask")
    # Historical element zero is the initial control state. The interval
    # starting at tick k receives k+1, or the last held target in the tail.
    expected = original[np.minimum(np.arange(ticks[-1]+1)+1, len(clock)-1)]
    equal(a["all_commands"], expected, path, "original command stream")
    equal(a["executed"], expected[ticks[:-1]], path, "applied joint action")
    equal(a["actuator"][:, a["actuator_indices"]], a["executed"], path, "actuator mapping")
    if "source_state_index" in a:
        native = array_dict(source/"run/robot_apple_timeline.npz")
        ix = a["source_state_index"]
        equal(t, native["timestamps"][ix], path, "robot source time")
        equal(a["q"], native["robot_qpos"][ix], path, "native measured position")
        equal(a["qd"], native["robot_qvel"][ix], path, "native measured velocity")
    if not np.isfinite(a["object"]).all() or not np.allclose(
        np.linalg.norm(a["object"][:, 3:7], axis=1), 1., rtol=0, atol=1e-6
    ):
        raise ValueError(f"{path}: invalid metric object pose")
    return a


def validate_phase_and_residual(row, a, cfg):
    path = row["path"]
    end = len(a["t"])
    for n in sorted(set([1, end//2, end-1, end, *(
        np.flatnonzero(np.diff(a["phase_id"])) + 2)])):
        # Every input past this boundary is removed, including future commands.
        prefix = {k: v[:n] if v.ndim and len(v) == end else v for k, v in a.items()}
        prefix["all_commands"] = a["all_commands"][:a["control_ticks"][n-1]+1]
        labels_prefix, result = labels(prefix, cfg["phase"], n == end)
        for k, v in labels_prefix.items():
            equal(v, a[k][:n], path, f"causal {k} at {n}")
        if result["success"] != (n == end and row["success"]):
            raise ValueError(f"{path}: incomplete suffix accepted or saved outcome differs")
    restart = Phase(cfg["phase"])
    restart.advance(0., obs_at(a, 0), False, stable_interval=False)
    outcome = restart.outcome(True)
    if outcome["stable_grasp"] or outcome["transported"] or outcome["success"]:
        raise ValueError(f"{path}: a reset carried acceptance over")
    if labels(a, cfg["phase"], True, continuous=False)[1]["success"]:
        raise ValueError(f"{path}: local reset qualified as a complete demonstration")
    limits = ActionLimits(*(a[k] for k in ("action_lo","action_hi","residual_scale","residual_max_step")))
    actions = a["all_commands"]
    hand, delta, _ = split_residual(actions[:, LEFT], limits, np.random.default_rng(42), "perturbed")
    expected = actions.copy()
    nominal = actions.copy()
    nominal[:, LEFT] = hand
    previous = np.zeros(7)
    worst = 0.
    for u, r, y in zip(nominal, delta, expected):
        restored, previous = limits.execute(u, r, previous)
        worst = max(worst, float(np.max(abs(restored-y))))
    if not np.any(hand != actions[:, LEFT]) or worst > 1e-7:
        raise ValueError(f"{path}: legal nominal/residual fails recomposition ({worst})")
    return worst


def verify(root):
    cfg = config()
    apple = read(root/"apple_summary.json")
    manifest = {r["episode"]: r for r in read(root/"apple_split.json")}
    sonic = read(root/"sonic_summary.json")
    groups = {s: set() for s in ("train", "val", "test")}
    for r in [*manifest.values(), *sonic["records"]]:
        groups[r["split"]].add(r["source_group_id"])
    split_map(groups)
    counts = {"sparse": Counter(), "dense": Counter(), "sonic": Counter()}
    dense = [read(p) for p in sorted((root/"dense").glob("apple_*/summary.json"))]
    for split in groups:
        if not any(r["identity"]["split"] == split for r in dense):
            raise ValueError(f"{split}: no completed 50 Hz replay")
    examples = {}
    for key, rows in (("sparse", apple["records"]), ("dense", dense)):
        for r in rows:
            a = validate_apple(r, manifest, cfg)
            counts[key][r["identity"]["split"]] += int(a["actual_20ms"].sum())
            kind = r["failure"] or "success"
            if (key, kind) not in examples:
                worst = validate_phase_and_residual(r, a, cfg)
                examples[key, kind] = {"file": r["path"], "recomposition_max_rad": worst}
    for r in sonic["records"]:
        a = array_dict(r["path"])
        if len(a["q"]) != r["samples"] or a["q"].shape[1:] != (43,) or not np.isfinite(a["q"]).all():
            raise ValueError(f"{r['path']}: inconsistent SONIC samples")
        counts["sonic"][r["split"]] += len(a["q"])
    return {"episodes": {"sparse": len(apple["records"]), "dense": len(dense), "sonic": len(sonic["records"])},
            "groups": {k: len(v) for k, v in groups.items()}, "transitions_20ms": {
                k: counts[k] for k in ("sparse", "dense")}, "sonic_state_samples": counts["sonic"],
            "prefix_reset_and_residual": {f"{k}:{v}": m for (k, v), m in examples.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    write(args.root/"verification.json", verify(args.root))

"""Batch normalization of genuine records, causal labeling and targeted replay."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time

import mujoco as mj
import numpy as np

from .contract import (EVENTS, GROUPS, LEFT, ROOT, SCHEMA, candidate, config, exact_indices,
                       extend_split, read, save_arrays, sha256, split_map, write)
from .phase import Phase, RULE_VERSION, phase_arrays
from .scene import NativeScene, Sensors, compact_model, stable_intervals


META = ("joint_names", "points", "physical", "physical_valid", "actuator_indices",
        "action_lo", "action_hi", "residual_scale", "residual_max_step")


def array_dict(path):
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def split_apple(cfg, ids=range(402)):
    split = split_map(read(cfg["teacher"]/"splits.json")["groups"])
    return extend_split(split, ids, cfg["seed"])


def obs_at(a, i):
    fields = ("q", "qd", "object", "twist", "wrist", "wrist_twist", "tips", "target", "dims",
              "diameter", "margin", "groups", "groups_valid", "original_support", "target_support",
              "robot_contact", "support_forces", "contact_counts")
    return {k: a[k][i] for k in fields}


def labels(a, rules, completed, continuous=True, error=None):
    t = a["t"]
    phase = Phase(rules)
    # Difference between current and preceding known nominal positions.
    # Offline always use the same real command chronology as online.
    source = a["all_commands"] if "all_commands" in a else a["executed"]
    t0 = .02 if "all_commands" in a else t[1]-t[0]
    kk = np.minimum(np.rint(t/t0).astype(int), len(source)-1)
    closing = closure(source[kk], source[np.maximum(kk-1, 0)], t0, rules)
    rows = [phase.advance(float(x), obs_at(a, i), bool(closing[i]),
                          stable_interval=bool(a["stable_interval"][i])) for i, x in enumerate(t)]
    out = phase_arrays(rows)
    out["events"] = np.c_[a["groups"], out["slip"], a["target_support"]]
    out["events_valid"] = np.c_[a["groups_valid"], out["phase_valid"], np.ones(len(t), dtype=bool)]
    return out, phase.outcome(completed, continuous, error)


def closure(current, previous, dt, rules):
    # Index/middle close along negative axes, distal thumb along positive.
    return np.mean((current[..., [29, 30, 31, 32, 35]]-previous[..., [29, 30, 31, 32, 35]]) *
                   np.array([-1., -1., -1., -1., 1.]), axis=-1) >= dt*rules["closure_rad_per_second"]


def raw_targets(c, cfg):
    canon = read(c/"scene/robot.json")
    names = canon["body_joint_names"]+canon["hand_joint_names"]
    run = c/"run"
    z = array_dict(run/"control_commands.npz")
    t = z["timestamps_50hz"]
    ctl = z["clipped_targets43"]
    if (len(names) != 43 or len(set(names)) != 43 or z["joint_names"].tolist() != names
            or ctl.shape != (len(t), 43) or not np.isfinite(ctl).all() or t[0] != 0
            or not np.allclose(np.diff(t), cfg["control_dt"], atol=1e-9, rtol=0)):
        raise ValueError(f"{c}: bad source control contract")
    return names, t, ctl


def finger_evidence(path, t, dt, saved_left, full, model):
    # Per-step source stores force evaluation at t_state - dt. Filter by group
    # before decoding (~98% of the verbose native contacts are unrelated).
    groups = np.zeros((len(t), 3), dtype=bool)
    bodies = [model.body(i).name for i in model.geom_bodyid]
    with gzip.open(path, "rb") as f:
        for line in f:
            if b'"apple_robot"' not in line:
                continue
            c = json.loads(line)
            if c["group"] != "apple_robot" or c["contact_force"][0] <= .001 or not c.get("active", True):
                continue
            i = c.get("physics_step_index", round((c["t"]+dt-t[0])/dt))
            if not 0 <= i < len(t) or abs(t[i]-dt-c["t"]) > 1e-8:
                raise ValueError(f"{path}: contact/state timestamp mismatch")
            pair = c["bodies"] if "bodies" in c else [bodies[g] for g in c["geom"] if g >= 0]
            for j, prefix in enumerate(GROUPS):
                if any(b.startswith(f"left_hand_{prefix}_") for b in pair):
                    groups[i, j] = True
    known = (groups.any(axis=1) == saved_left) & (full | ~saved_left)
    return groups, np.repeat(known[:, None], 3, axis=1)


def package(rows, s, times, commands, stable, cfg, c, limits=None, ids=None):
    a = {k: np.asarray([x[k] for x in rows]) for k in rows[0]}
    if limits is None:
        jid = np.array([s.m.joint(n).id for n in s.names])
        ids = np.array([s.m.actuator(n).id for n in s.names])
        from .contract import ActionLimits
        from types import SimpleNamespace
        sim = SimpleNamespace(joint_limits=s.m.jnt_range[jid], model=s.m, actuator_ids=ids)
        limits = ActionLimits.from_scene(SimpleNamespace(sim=sim), cfg["residual_scale_fraction"], cfg["control_dt"])
    a.update(t=times, joint_names=np.asarray(s.names), points=s.point_cloud, physical=s.physical,
             physical_valid=s.physical_valid, stable_interval=stable,
             actuator_indices=ids, action_lo=limits.lo, action_hi=limits.hi,
             residual_scale=limits.scale, residual_max_step=limits.max_step, all_commands=commands,
             nominal_wrist=s.wrist_plan(commands[:, :29]),
             schema=np.array(SCHEMA), rule_version=np.array(RULE_VERSION))
    return a


def sparse_episode(c, cfg):
    names, tctrl, ctl = raw_targets(c, cfg)
    z = array_dict(c/"run/robot_apple_timeline.npz")
    if z["joint_names"].tolist() != names:
        raise ValueError(f"{c}: observation and command joint orders differ")
    t0 = cfg["control_dt"]
    ticks = np.rint(z["timestamps"]/t0).astype(int)
    ii = np.flatnonzero((ticks >= 0) & np.isclose(ticks*t0, z["timestamps"], atol=1e-8, rtol=0))
    tt, ticks = z["timestamps"][ii], ticks[ii]
    if len(ii) < 2 or ticks[0] != 0 or np.any(np.diff(ticks) > 5):
        raise ValueError(f"{c}: exact-matched sparse state stream is incomplete")
    r = read(c/"run/report.json")
    ev = read(c/"evaluation.json")
    raw = array_dict(c/"run/metrics.npz")
    phys_t, dt = raw["t"], cfg["physics_dt"]
    if r["dt_s"] != dt or raw["count_names"][:4].tolist() != ["apple_pad", "apple_rigid_table", "apple_plate", "apple_robot"]:
        raise ValueError(f"{c}: native metric schema differs")
    jj = exact_indices(phys_t, tt, c/"run/metrics.npz")
    if not (np.array_equal(z["apple_qpos"][ii], raw["apple_qpos"][jj]) and
            np.array_equal(z["apple_qvel"][ii], raw["apple_qvel"][jj])):
        raise ValueError(f"{c}: original robot and object clocks disagree")
    # Every actual command in [t_i,t_i+1), not a fabricated single action.
    # The first target in the old file is initialization, not tick zero.
    prev = ctl[np.minimum(ticks, len(ctl)-1)]
    if not np.array_equal(prev[1:], z["robot_ctrl"][ii[1:]]):
        raise ValueError(f"{c}: held target is not the command that reached the logged state")
    active_stream = ctl[np.minimum(np.arange(ticks[-1]+1)+1, len(ctl)-1)]
    if any(not np.isfinite(z[k][ii]).all() for k in ("robot_qpos", "robot_qvel", "robot_ctrl", "apple_qpos", "apple_qvel")):
        raise ValueError(f"{c}: nonfinite actual observation")
    s = Sensors(compact_model(c/"scene/scene.xml"), names, c, ev["evaluation_config"])
    g, known = finger_evidence(c/"run/contacts.jsonl.gz", phys_t, dt, raw["left_finger_contact"],
                               r["contact_recording"]["mode"] == "every_physics_step", s.m)
    u = np.c_[raw["table_force"][:, 2], raw["plate_force"][:, 2]]
    obs = [s.observation(z["robot_qpos"][i], z["robot_qvel"][i], z["apple_qpos"][i], z["apple_qvel"][i],
                         s.contact_obs(g[j], raw["counts"][j, :4], u[j], known[j])) for i,j in zip(ii,jj)]
    stable = stable_intervals(tt, phys_t, raw["apple_qpos"], raw["apple_qvel"], raw["counts"], u[:, 1], u[:, 0],
                              s.geo, ev["evaluation_config"], dt)
    a = package(obs, s, tt, active_stream, stable, cfg, c)
    a.update(control_ticks=ticks, source_state_index=ii, executed=active_stream[ticks].copy(),
             initial_target=z["robot_ctrl"][ii[0]], sampled_prev_target=z["robot_ctrl"][ii].copy(),
             actuator=active_stream[ticks[:-1]][:, np.argsort(a["actuator_indices"])],
             motion_end_s=tctrl[-1])
    a["executed"] = a["executed"][:-1]  # no intervals beyond the recorded last state
    a["command_start_s"], a["command_end_s"] = tt[:-1], tt[1:]
    a["actual_20ms"] = np.isclose(np.diff(tt), t0, rtol=0, atol=1e-8)
    out, result = labels(a, cfg["phase"], r["completed"] and r["tail_completed"])
    a.update(out)
    return a, dict(result, native_accepted=ev["accepted"], native_physics_valid=ev["physics_valid"],
                   group_observation_fraction=float(known[jj].mean()),
                   episode_source=c, mechanics="archived actual states; all_commands has every 50 Hz held setpoint",
                   motion_end_s=float(tctrl[-1]), physical_samples=len(phys_t), recorded_state_samples=len(tt))


def index_apple(cfg):
    previous = split_apple(cfg)
    root = cfg["replay"]
    raw = read(root/"baseline_results.json")
    rows = []
    for row in raw:
        i = int(row["episode"])
        c = candidate(cfg, i)
        e = row["evaluation"] or {}
        required = [c/"run"/n for n in ("report.json", "robot_apple_timeline.npz", "metrics.npz",
                                         "control_commands.npz", "contacts.jsonl.gz")]
        complete = e.get("completed", False) and all(x.is_file() for x in required)
        rows.append(dict(episode=i, source_group_id=f"apple-d89c126a-{i:06d}", reset_id=0,
                         split=previous[i], source=c, teacher=bool(e.get("accepted", False)), usable=complete,
                         excluded=None if complete else "incomplete_native_or_missing_records"))
    return rows


def _sparse_worker(task):
    cfg, root, row = task
    i = row["episode"]
    out = root/"sparse"/f"apple_{i:06d}"
    identity = dict(row, rule_version=RULE_VERSION, rules=cfg["phase"],
                    source_report_sha256=sha256(row["source"]/"run/report.json"))
    meta = out.with_suffix(".json")
    dst = out.with_suffix(".npz")
    if meta.is_file():
        d = read(meta)
        if not dst.is_file() or d["identity"] != json.loads(json.dumps(identity, default=str)):
            raise ValueError(f"{out}: resume identity changed")
        return d
    start = time.monotonic()
    a, result = sparse_episode(row["source"], cfg)
    save_arrays(dst, **a)
    data = dict(identity=identity, path=dst, **result, sample_count=len(a["t"]),
                actual_20ms=int(a["actual_20ms"].sum()),
                phase_counts=np.bincount(a["phase_id"], minlength=5)[1:], valid_phase=int(a["phase_valid"].sum()),
                wall_seconds=time.monotonic()-start)
    write(meta, data)
    print(f"apple {i:03d} {result['success']=} {result['failure']=} n={len(a['t'])} ({time.monotonic()-start:.1f}s)", flush=True)
    return data


def audit_apple(cfg, out, ids, workers):
    rows = index_apple(cfg)
    write(out/"apple_split.json", rows)
    rows = [x for x in rows if ids is None or x["episode"] in ids]
    if not rows:
        raise ValueError("No selected episodes")
    tasks = [(cfg, out, x) for x in rows if x["usable"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_sparse_worker, tasks))
    summaries = {s: [r for r in results if r["identity"]["split"] == s] for s in ("train", "val", "test")}
    write(out/"apple_summary.json", dict(rows=len(rows), exclusions=[x for x in rows if not x["usable"]],
        split={s: dict(episodes=len(rs), observations=sum(r["sample_count"] for r in rs),
            actual_20ms=sum(r["actual_20ms"] for r in rs), success=sum(r["success"] for r in rs),
            failures=dict(Counter(r["failure"] for r in rs if r["failure"])),
            phases=np.sum([r["phase_counts"] for r in rs], axis=0) if rs else [0]*4)
            for s,rs in summaries.items()}, records=results, event_fields=EVENTS))


def dense_replay(cfg, out, episode, seconds, checkpoint_every):
    c = candidate(cfg, episode)
    names, t, ctl = raw_targets(c, cfg)
    sim = NativeScene(c, cfg)
    if names != sim.sim.joint_names:
        raise ValueError(f"{c}: compiled order differs")
    clock = cfg["control_dt"]
    total = round((sim.end+sim.tail)/clock) if seconds is None else round(seconds/clock)
    if not 0 < total <= round((sim.end+sim.tail)/clock):
        raise ValueError("Requested duration outside native task/tail")
    tt = np.arange(total+1)*clock
    stream = ctl[np.minimum(np.arange(total+1)+1, len(ctl)-1)]
    ident = dict(episode=episode, source_group_id=f"apple-d89c126a-{episode:06d}", reset_id=0,
                 split=split_apple(cfg)[episode], original=c, schema=SCHEMA, rules=cfg["phase"],
                 steps=total, assets=sim.assets(), source_report_sha256=sha256(c/"run/report.json"))
    root = out/f"apple_{episode:06d}"
    partial = root/"resume.npz"
    done = root/"summary.json"
    write(root/"request.json", ident)
    if done.exists():
        if read(done)["identity"] != read(root/"request.json"):
            raise ValueError(f"{root}: completed replay has different identity")
        return
    if partial.exists():
        z = array_dict(partial)
        if json.loads(str(z.pop("identity_json"))) != read(root/"request.json"):
            raise ValueError(f"{root}: replay resume requires identical inputs")
        sim.restore(z.pop("integration"), resume=True)
        rows = [{k: column[i] for k, column in z.items() if k != "stable_interval"} for i in range(len(z["q"]))]
        stable = list(z["stable_interval"])
        initial = sim.restore_original(0.) if len(rows) == 0 else array_dict(root/"initial.npz")["integration"]
    else:
        initial = sim.restore_original(0.)
        # mj_setState does not restore the previous solve's contacts.
        # For the exact original t=0 reset recover it from native evidence.
        raw = array_dict(c/"run/metrics.npz")
        k = exact_indices(raw["t"], [0.], c)[0]
        if raw["left_finger_contact"][k]:
            groups, known = finger_evidence(c/"run/contacts.jsonl.gz", raw["t"], sim.dt,
                raw["left_finger_contact"], sim.report["contact_recording"]["mode"] == "every_physics_step", sim.m)
            g, gv = groups[k], known[k]
        else:
            g, gv = np.zeros(3, dtype=bool), np.ones(3, dtype=bool)
        s = sim.sensors
        evidence = s.contact_obs(g, raw["counts"][k, :4], np.array([raw["table_force"][k, 2], raw["plate_force"][k, 2]]), gv)
        rows = [s.observation(*sim.physical_state(), evidence)]
        stable = [False]
        save_arrays(root/"initial.npz", integration=initial)
    wall = time.monotonic()
    for i in range(len(rows)-1, total):
        prev = rows[-1]
        contact, success_interval = sim.step(stream[i], observe_stability=bool(prev["margin"] > -.012
                          and (prev["object"][2]-prev["target"][2]) < .15))
        rows.append(sim.observe(contact))
        stable.append(success_interval)
        if (i+1) % checkpoint_every == 0 or i+1 == total:
            save_arrays(partial, integration=sim.snapshot(), identity_json=np.array((root/"request.json").read_text()),
                        stable_interval=np.asarray(stable), **{k: np.asarray([x[k] for x in rows]) for k in rows[0]})
            print(f"replay {episode:03d} {i+1}/{total} t={float(sim.d.time):.2f}s wall={time.monotonic()-wall:.0f}s", flush=True)
    s = sim.sensors
    a = package(rows, s, tt, stream, np.asarray(stable), cfg, c, sim.limits, sim.sim.actuator_ids)
    z = array_dict(c/"run/robot_apple_timeline.npz")
    ticks = np.rint(z["timestamps"]/clock).astype(int)
    ix = np.flatnonzero((ticks >= 0) & (ticks <= total) & np.isclose(ticks*clock, z["timestamps"], atol=1e-8, rtol=0))
    ids = ticks[ix]
    pw = np.asarray([s.base.apply(x[:3])+s.origin for x in a["object"][ids]])
    max_robot_error = float(np.max(abs(a["q"][ids]-z["robot_qpos"][ix])))
    max_object_error = float(np.max(abs(pw-z["apple_qpos"][ix, :3])))
    # Measurements remain genuine even if engine versions differ, but an
    # unverified reproduction cannot stand in for the historical trajectory.
    equivalent = max_robot_error < 1e-5 and max_object_error < 1e-5
    a.update(executed=stream[:-1], actuator=np.broadcast_to(stream[:-1, np.argsort(sim.sim.actuator_ids)], (total, 43)),
             command_start_s=tt[:-1], command_end_s=tt[1:], actual_20ms=np.ones(total, bool),
             control_ticks=np.arange(total+1), motion_end_s=sim.end,
             initial_target=z["robot_ctrl"][ix[0]])
    out_label, result = labels(a, cfg["phase"], len(rows) == round((sim.end+sim.tail)/clock)+1)
    a.update(out_label)
    save_arrays(root/"trajectory.npz", **a)
    write(done, dict(identity=read(root/"request.json"), path=root/"trajectory.npz", **result,
                     native_accepted=read(c/"evaluation.json")["accepted"],
                     native_physics_valid=read(c/"evaluation.json")["physics_valid"],
                     sparse_comparison=dict(matched=len(ix), max_joint_rad=max_robot_error,
                                            max_object_m=max_object_error, equivalent=equivalent),
                     timesteps=total, wall_seconds=time.monotonic()-wall))
    partial.unlink()


def sonic_groups(items):
    # Multiple slices from the same video and duplicate published IDs belong
    # together even if different reference names appear in the archives.
    parents = {}
    def parent(k):
        while k in parents:
            k = parents[k]
        return k
    keys = []
    for x in items:
        m = re.match(r"^([A-Za-z0-9_-]{11})_\d+_\d+_\d+$", x["sample_name"])
        m = m or re.search(r"YouTube-Dance_movies_([A-Za-z0-9_-]{11})_\d+$", x["sample_name"])
        a = "sonic-id:"+x["sample_id"]
        b = "sonic-video:"+m[1] if m else a
        aa, bb = sorted((parent(a), parent(b)))
        if aa != bb:
            parents[bb] = aa
        keys.append(a)
    return [parent(k) for k in keys]


def read_sonic(cfg, out):
    root = cfg["sonic"]
    ref = cfg["sonic_xml"]
    m = mj.MjModel.from_xml_path(str(ref))
    names, _, _ = raw_targets(candidate(cfg, 1), cfg)
    if m.nq != 50 or m.njnt != 44 or any(m.joint(n).type[0] != mj.mjtJoint.mjJNT_HINGE for n in names):
        raise ValueError(f"{ref}: tracking qpos name mapping differs")
    indices = np.asarray([m.joint(n).qposadr[0] for n in names])
    items = [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
    groups = sonic_groups(items)
    split = extend_split({}, groups, cfg["seed"])
    records, exclusions = [], []
    by_archive = {}
    for row, key in zip(items, groups):
        by_archive.setdefault(row["archive"], []).append((row, key))
    for archive, rows in sorted(by_archive.items()):
        with tarfile.open(root/Path(archive).name) as tar:
            for record, group in rows:
                member = record["files"]["g1_tracking"]
                with tar.extractfile(member) as fp:
                    source = fp.read()
                if hashlib.sha256(source).hexdigest() != record["sha256"]["g1_tracking"]:
                    raise ValueError(f"{archive}:{member}: source checksum mismatch")
                with np.load(io.BytesIO(source), allow_pickle=False) as z:
                    x, t = z["qpos"], z["times"]
                if x.shape != (len(t), m.nq) or not np.isfinite(x).all():
                    exclusions.append(dict(archive=archive, member=member, reason="invalid_qpos"))
                    continue
                want = np.arange(round((t[-1]-t[0])/cfg["control_dt"])+1)*cfg["control_dt"]+t[0]
                want = want[want <= t[-1]+1e-9]
                try:
                    ii = exact_indices(t, want, member)
                except ValueError as exc:
                    exclusions.append(dict(archive=archive, member=member, reason=str(exc)))
                    continue
                if len(ii) < 2:
                    exclusions.append(dict(archive=archive, member=member, reason="short_recording"))
                    continue
                # qvel is not archived. State-only warmstart uses joint
                # positions and optional backward position differences, never
                # these differences as measured velocity or executed control.
                f = out/"sonic"/(record["sample_name"]+".npz")
                save_arrays(f, t=t[ii], q=x[ii[:, None], indices].astype(np.float32))
                records.append(dict(path=f, source_group_id=group, split=split[group], samples=len(ii),
                                    archive=root/Path(archive).name, source_member=member))
        print(f"sonic {archive} tracked records={len(records)} exclusions={len(exclusions)}", flush=True)
    write(out/"sonic_summary.json", dict(records=records, exclusions=exclusions, joint_names=names, qpos_indices=indices,
          source_manifest_sha256=sha256(root/"manifest.jsonl"), scene_xml=ref, scene_xml_sha256=sha256(ref),
          supervision="state-only auxiliary; no executed commands, simulator velocities, objects or events in source"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT/"configs/experiment.json")
    sub = p.add_subparsers(dest="command", required=True)
    pa = sub.add_parser("apple")
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--episodes", type=int, nargs="+")
    pa.add_argument("--workers", type=int, default=2)
    pr = sub.add_parser("replay")
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--episode", type=int, required=True)
    pr.add_argument("--seconds", type=float)
    pr.add_argument("--checkpoint-every", type=int, default=100)
    ps = sub.add_parser("sonic")
    ps.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cfg, out = config(a.config), a.out.resolve()
    if a.command == "apple":
        audit_apple(cfg, out, a.episodes, a.workers)
    elif a.command == "replay":
        dense_replay(cfg, out, a.episode, a.seconds, a.checkpoint_every)
    else:
        read_sonic(cfg, out)


if __name__ == "__main__":
    main()

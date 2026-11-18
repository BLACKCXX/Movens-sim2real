"""Audit closed-loop logs and export continuous training successes to SimWindows."""
import argparse
import fcntl
from pathlib import Path

import numpy as np
from PIL import Image

from movens_interact.normalization import MotionNormalizer
from movens_interact.root40 import recover_motion_abs_from_root40_action
from training.paired_data import SimWindows, robot_model_info, rollout_window_data

from .contract import ActionLimits, ROOT, SCHEMA, candidate, config, read, save_arrays, sha256, write
from .data import array_dict, closure, obs_at, split_apple
from .phase import Phase, RULE_VERSION, phase_arrays


def audit(path):
    row = read(path)
    identity = row["identity"]
    root = Path(path).parent
    a = array_dict(row["path"])
    p = array_dict(root/"plans.npz")
    dt, n = identity["dt"], len(a["executed"])
    active = round(float(a["motion_end_s"])/dt)
    i = np.arange(n)
    if (identity["schema"] != SCHEMA or identity["rule_version"] != RULE_VERSION
            or not row["reached_protocol_initial"] or not bool(a["continuous_reset"])
            or row["decision_steps"] != n or not n or identity["dt"] != .02 or identity["physics_dt"] != .00025
            or len(a["t"]) != n+1 or not np.array_equal(a["t"], np.arange(n+1)*dt)
            or not np.array_equal(a["tick"], i) or not np.array_equal(a["t"], np.r_[a["t_command"], a["t_command_end"][-1]])
            or not np.array_equal(a["t"][1:],a["t_command_end"]) or not np.all(a["actual_20ms"])
            or not np.array_equal(a["actuator"][:, a["actuator_indices"]], a["executed"])
            or not np.array_equal(a["decision_mask"], i < active)
            or any(np.any(a[k][:-1]) for k in ("terminated","truncated"))):
        raise ValueError(f"{path}: invalid command/state/reset contract")
    if row["engineering_error"]:
        raise ValueError(f"{path}: engineering-invalid rollout")
    if bool(a["terminated"][-1]) != row["complete"] or bool(a["truncated"][-1]) == row["complete"]:
        raise ValueError(f"{path}: inconsistent terminal/collection state")
    z = array_dict(root/"initial.npz")
    if not np.array_equal(z["actuator"][a["actuator_indices"]],a["initial_target"]):
        raise ValueError(f"{path}: initial actuator state differs")
    if (not np.array_equal(p["plan_id"], np.arange(len(p["plan_id"])))
            or not np.array_equal(p["first_physical_frame"], 1+20*p["plan_id"])
            or not np.array_equal(p["prefix_length"], np.where(p["plan_id"] == 0, 0, 10))):
        raise ValueError(f"{path}: action chunk, first command or committed prefix differs")
    plan_i = np.minimum(np.arange(n+1),active-1)
    nominal = np.c_[p["body_absolute"][plan_i//20,plan_i%20,:29],
                    p["hand_absolute"][plan_i//20,plan_i%20]].astype(np.float64)
    if (not np.array_equal(nominal[:-1],a["nominal"])
            or not np.array_equal(plan_i[:-1]//20,a["plan_id"])
            or not np.array_equal(plan_i[:-1]%20,a["plan_offset"])):
        raise ValueError(f"{path}: actual nominal commands do not match decoded action chunks")
    limits=ActionLimits(*(a[k] for k in ("action_lo","action_hi","residual_scale","residual_max_step")))
    correction=np.zeros(7)
    executed=a["initial_target"]
    phase=Phase(identity["rules"])
    phases=[]
    for k,t in enumerate(a["t"]):
        prev = nominal[k-1] if k else a["initial_target"]
        close = bool(closure(nominal[k],prev,dt,identity["rules"]))
        phases.append(phase.advance(t,obs_at(a,k),close,stable_interval=bool(a["stable_interval"][k])))
        if k < n:
            if k < active:
                executed,correction=limits.execute(nominal[k],a["selected_residual"][k],correction)
            if (not np.array_equal(executed,a["executed"][k])
                    or not np.array_equal(correction,a["executed_correction"][k])
                    or not np.array_equal(a["residual_proposal"][k],a["selected_residual"][k])
                    or (not identity["residual_enabled"] and np.any(a["residual_proposal"][k]))):
                raise ValueError(f"{path}: D0 target recomposition differs at tick {k}")
    out=phase.outcome(row["complete"])
    if any(out[k] != row[k] for k in out) or any(not np.array_equal(v,a[k]) for k,v in phase_arrays(phases).items()):
        raise ValueError(f"{path}: acceptance or phase does not match recorded physical evidence")
    return row,a,p


def paired(a, identity):
    rgb = a["rgb_ticks"]
    t = a["t"][rgb]
    if (not np.array_equal(t,a["rgb_time_s"]) or rgb[0] != 0
            or not np.array_equal(rgb,np.arange(0,len(a["t"]),5))):
        raise ValueError("Expected actual causal RGB at exact 10 Hz state boundaries")
    root = identity["asset"]["base_xyz_wxyz"]
    q = a["q"][rgb]
    return rollout_window_data(dict(timestamp_s=t,control_timestamp_s=a["t_command"],
        control_targets_rad=a["executed"],body_obs_abs36=np.c_[q[:,:29],np.broadcast_to(root,(len(q),7))],
        root_pose_wxyz=np.asarray([root]),hand_obs14=q[:,29:],frame_index=np.arange(len(rgb))),
        task_end=float(a["motion_end_s"]))


def verify_windows(index, expected, body, hand, loader=SimWindows):
    data = loader(index,None,body,hand,"train",stride=10)
    max_body = max_hand = 0.
    for k,(slot,j) in enumerate(data.windows):
        item = data[k]
        e = data.episodes[slot]
        commands = expected[item["episode_index"]]
        valid = item["valid"]
        ref = commands[item["target_control_indices"][valid]-1]
        state = e["body_obs_abs"][j:j+1]
        ext = recover_motion_abs_from_root40_action(body.denormalize_action(item["body_action"]),state)
        max_body=max(max_body,float(np.max(abs(ext[valid,:29]-ref[:,:29]))))
        max_hand=max(max_hand,float(np.max(abs(hand.denormalize_action(item["hand_action"])[valid]-ref[:,29:]))))
        if (not np.allclose(ext[:,29:32],state[:,29:32],rtol=0,atol=1e-6)
                or item["rgb_timestamp"] > item["timestamp"] + 1e-9):
            raise ValueError("Fixed-base reference or causal context changed")
    if max(max_body,max_hand)>1e-6:
        raise ValueError(f"Original train codec changed the actual commands: {max_body=}, {max_hand=}")
    return dict(windows=len(data),body29_roundtrip_max_rad=max_body,hand14_roundtrip_max_rad=max_hand)


def finish_export(out, template, cfg, rows, excluded, expected, body_source, *, domain="mujoco", loader=SimWindows):
    write(out/"selection.json",dict(selected=rows,excluded=excluded))
    if not rows:
        raise ValueError(f"No eligible continuous training success; see {out/'selection.json'}")
    index=out/"index.json"
    fields=("format","data_revision","fps","scene_xml","scene_sha256","kinematics_sha256","body_joint_names","hand_joint_names")
    write(index,dict({k:template[k] for k in fields},episodes=rows,collection_complete=True,domain=domain,
        body_reference_source=body_source,hand_reference_source="actual applied Hand14 after residual and limits",ego_fps=10))
    body,hand=[MotionNormalizer.from_file(cfg["teacher"]/template[k+"_normalization"]) for k in ("body","hand")]
    write(out/"verification.json",verify_windows(index,expected,body,hand,loader))


def export(out, paths, cfg):
    template = read(cfg["teacher"]/"index.json")
    split = split_apple(cfg)
    rows, excluded, expected = [], [], {}
    # Stable local IDs are separate from original upstream episode IDs.
    # The training split is still assigned by the original common source.
    for path in sorted(set(p.resolve() for p in paths)):
        r = read(path)
        identity = r["identity"]
        ep = identity["episode"]
        if (identity["split"] != split[ep] or identity["source_group_id"] != f"apple-d89c126a-{ep:06d}"
                or identity["task"] != cfg["task"] or identity["rules"] != cfg["phase"]):
            raise ValueError(f"{path}: source split, phase rules or text changed")
        if not (r["success_export_eligible"] and r["complete"] and r["success"] and split[ep]=="train"):
            excluded.append(dict(path=path,split=split[ep],reason=r["failure"] or "held_out_source"))
            continue
        r,a,p = audit(path)
        work = path.parent
        d = paired(a,identity)
        if int(a["decision_mask"].sum()) != len(d["timestamps"])-1:
            raise ValueError(f"{path}: action count changed in export")
        xml = Path(identity["asset"]["xml"])
        c = candidate(cfg,ep)
        if xml != c/"scene/scene.xml" or identity["asset"]["xml_sha256"] != sha256(xml):
            raise ValueError(f"{path}: simulator scene identity changed")
        body,hand = template["body_joint_names"],template["hand_joint_names"]
        if a["joint_names"].tolist() != body+hand:
            raise ValueError(f"{path}: canonical robot contract changed")
        robot = robot_model_info(xml,body,hand)
        if robot["kinematics_sha256"] != template["kinematics_sha256"]:
            raise ValueError(f"{path}: robot incompatible with original trainer")
        init = array_dict(work/"initial.npz")
        if not np.array_equal(init["qpos"][robot["qpos_indices"]],a["q"][0]):
            raise ValueError(f"{path}: state zero is detached from recorded task reset")
        eid = 1_000_000+len(rows)
        data_path, rgb_path = (Path(folder)/f"episode_{eid:06d}.npz" for folder in ("episodes","rgb"))
        # Tail RGB/commands remain in the original validated log; only task
        # actions train the policy. Original 40D plan and anchor stay available.
        frame_ids = d["window_video_frame_indices"]
        frames = np.stack([np.array(Image.open(work/"rgb"/f"{a['rgb_ticks'][j]:06d}.png")) for j in frame_ids])
        if frames.dtype != np.uint8 or frames.shape[1:] != (480,640,3):
            raise ValueError(f"{path}: original cache expects RGB uint8 640x480")
        d.update(rgb_timestamps=d["observation_timestamps"],rgb_source_frame_indices=np.arange(len(frames)),
                 nominal_body40_chunks=p["body_raw"],nominal_body40_anchor=p["anchor"],
                 nominal_hand14_chunks=p["hand_absolute"],task_text=np.array(identity["task"]),
                 applied_target43=a["executed"][:len(d["timestamps"])-1],
                 actuator43=a["actuator"][:len(d["timestamps"])-1],source_group_id=np.array(identity["source_group_id"]))
        save_arrays(out/data_path,**d)
        save_arrays(out/rgb_path,rgb=frames,timestamps=d["rgb_timestamps"],frame_indices=d["rgb_source_frame_indices"])
        rows.append(dict(episode_index=eid,original_episode_index=ep,source_group_id=identity["source_group_id"],
            reset_id=identity["reset_id"],trajectory=str(data_path),trajectory_sha256=sha256(out/data_path),
            rgb=str(rgb_path),rgb_sha256=sha256(out/rgb_path),scene_xml=str(xml),scene_sha256=identity["asset"]["xml_sha256"],
            kinematics_sha256=robot["kinematics_sha256"],split="train",success=True,teacher_success=True,rgb_ready=True,
            root_source="sim_known_fixed_base",frames=len(d["timestamps"]),observation_frames=len(frames),
            source_rollout=str(path),source_rollout_sha256=sha256(path),source_plans_sha256=sha256(work/"plans.npz")))
        expected[eid]=d["applied_target43"]
    finish_export(out,template,cfg,rows,excluded,expected,
                  "actual applied Body29; fixed simulated root; original nominal Body40 retained")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts",type=Path,nargs="+")
    p.add_argument("--config",type=Path,default=ROOT/"configs/experiment.json")
    p.add_argument("--out",type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--audit-only",action="store_true")
    mode.add_argument("--hardware",action="store_true",help="Export eligible independent hardware trial receipts")
    args=p.parse_args()
    if args.audit_only:
        for path in args.receipts:
            r,a,_=audit(path)
            print(path,"steps",len(a["executed"]),"complete",r["complete"],"success",r["success"])
    elif args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        with (args.out/".export.lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.hardware:
                from .real_export import export_hardware
                export_hardware(args.out,args.receipts,config(args.config))
            else:
                export(args.out,args.receipts,config(args.config))
    else:
        p.error("--out is required for export")


if __name__=="__main__":
    main()

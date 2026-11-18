"""Run the 50 Hz proposal path on original 30 Hz hardware captures, without DDS."""
import argparse
from pathlib import Path

import numpy as np

from movens_sim2real.contract import config, read, save_arrays, sha256, write, wxyz
from movens_sim2real.hardware import FixedBaseFK, Latest, WireOrder, nominal_from_capture, observation
from movens_sim2real.object_info import physical, read_task
from movens_sim2real.real_replay import load_capture
from movens_sim2real.robot_policy import RobotPolicy
from movens_sim2real.rollout import PLAN_ARRAYS, policy_load
from movens_sim2real.scene import compact_model


def run(a):
    cfg=config()
    task=read_task(a.task_info,cfg)
    c,source,parquet,video,_,names,q,t,pts,images=load_capture(cfg,a.episode,a.frames)
    model=compact_model(c/"scene/scene.xml")
    fk=FixedBaseFK(model,names)
    root=np.r_[fk.origin,wxyz(fk.axes.inv())]
    meta=read(a.calibration)
    if not meta.get("source"):
        raise ValueError("A named actuator calibration source is required")
    wire=WireOrder.from_calibration(names,meta)
    priors=task["priors"]
    phys=physical(priors)
    planner=policy_load(cfg,a.device)
    policy=RobotPolicy(cfg,a.small,names,wire,a.device,offline=True)
    streams=[Latest(.06),Latest(.12)]
    # Feed native captures only when they actually become available. Reusing a
    # recent measurement between captures is not a new state or transition.
    t0=max(t[0],pts[0])
    horizon=min(t[len(images)-1],pts[len(images)-1])
    ts=t0+np.arange(int(np.floor((horizon-t0)/.02))+1)*.02
    k=[0,0]
    records,plans=[],[]
    projection=None
    for tick,now in enumerate(ts):
        for stream,values,times,idx in zip(streams,(q,images),(t,pts),range(2)):
            while k[idx]<len(images) and times[k[idx]]<=now:
                value=values[k[idx]]
                stream.put(times[k[idx]],dict(q=value) if idx==0 else value)
                k[idx]+=1
        joint,camera=[stream.read(now) for stream in streams]
        ob,cloud=observation(joint,None,fk,priors,now)
        if projection is None:
            projection=wire.signs*ob["q"][wire.permutation]+wire.zero_rad
        block=None
        if tick%20==0:
            if tick==20:
                try:
                    policy.decide(now,tick,ob,cloud,phys,projection,task_left_s=max(0.,t[-1]-now))
                except ValueError as err:
                    if "not available" not in str(err):
                        raise
                else:
                    raise ValueError("A missing planner block generated a robot proposal")
            block=nominal_from_capture(planner,fk,joint,(camera[0],camera[1]),root,tick,now,cfg["task"],cfg["seed"],
                                       plans[-1] if plans else None)
            plans.append(block[0])
        # Plan readiness here is the logical replay time; actual inference
        # duration is retained separately and is not a realtime measurement.
        command=policy.decide(now,tick,ob,cloud,phys,projection,block=block,ready_s=now if block else None,
                              task_left_s=max(0.,t[-1]-now))
        proposed=command["sdk_target43"]
        diagnostic=policy.diagnostics(command,proposed,.02)
        unaffected=np.r_[0:29,36:43]
        if (np.any(abs(proposed-projection)>wire.speed_rad_s*.02+1e-8)
                or np.any((proposed<wire.lo_rad)|(proposed>wire.hi_rad))
                or not np.array_equal(command["issued_canonical43"][unaffected],command["limited_nominal"][unaffected])
                or command["dynamics_frame"][1,43:144].any() or command["phase"][1]
                or not all(np.isfinite(v).all() for v in diagnostic.values())):
            raise ValueError("Bounded command, absent-evidence masks or D0/critic outputs are inconsistent")
        projection=proposed
        records.append(dict(**{k:command[k] for k in ("time_s","tick","plan_id","plan_offset","nominal",
            "actor_proposal","limited_nominal","issued_canonical43","sdk_target43","actor_frame","dynamics_frame")},
            q=ob["q"],q_capture_s=joint[1],rgb_capture_s=camera[1],**diagnostic))
    try:
        policy.decide(ts[-1]+.02,len(ts),dict(ob,q_time_s=ts[-1]-.061),cloud,phys,projection,task_left_s=0.)
    except ValueError as err:
        if "fresh encoder" not in str(err):
            raise
    else:
        raise ValueError("A stale encoder sample generated a robot proposal")
    if len(plans)<2 or not np.any(np.stack([r["issued_canonical43"]!=r["limited_nominal"] for r in records])):
        raise ValueError("Probe needs a committed block refresh and a change in a hand target")
    save_arrays(a.out/"proposals.npz",**{k:np.asarray([r[k] for r in records]) for k in records[0]},
                joint_names=np.asarray(names))
    save_arrays(a.out/"plans.npz",**{k:np.asarray([r[k] for r in plans]) for k in PLAN_ARRAYS})
    write(a.out/"verification.json",dict(source="original AppleToPlate robot RGB and encoders",source_revision=source["revision"],
          inputs={str(p):sha256(p) for p in (parquet,video,a.small,a.calibration,a.task_info)},
          calibration_source=meta["source"],native_captures=len(np.unique([r["q_capture_s"] for r in records])),
          scheduled_proposals=len(records),plans=len(plans),max_encoder_age_s=float(np.max(ts-np.asarray([r["q_capture_s"] for r in records]))),
          planner_inference_s=[r["total_inference_seconds"] for r in plans],stale_encoder_and_missing_block_rejected=True,
          projected_position_and_rate_limits_verified=True,closed_loop=False,hardware_command_sent=False,
          applied_targets=0,physical_transitions=0,d0_inputs="bounded proposals only"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--small",type=Path,required=True)
    p.add_argument("--task-info",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--episode",type=int,default=1)
    p.add_argument("--frames",type=int,default=33)
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    if a.frames<14:
        p.error("Need native captures through a chunk refresh")
    run(a)

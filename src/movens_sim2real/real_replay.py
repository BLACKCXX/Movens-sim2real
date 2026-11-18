"""Check the deployment input boundary on original AppleToPlate robot recordings."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from training.image_sources import video_pts
from training.prepare_data import load_episode_reference

from .contract import ROOT, candidate, config, read, save_arrays, sha256, write, wxyz
from .features import dyn_frame, window
from .hardware import AppleMask, FixedBaseFK, Latest, Sam3Mask, hardware_features, nominal_from_capture, observation
from .object_info import physical, prepare, read_task
from .phase import Phase
from .rgbd import DepthObservations, recorded_depth
from .scene import compact_model


def load_capture(cfg,ep,frames):
    c=candidate(cfg,ep)
    source=read(c/"scene/prepared_manifest.json")
    parquet,video=(Path(source[k]) for k in ("source_parquet","source_rgb"))
    a,mapping=load_episode_reference(parquet,ep)
    # Use the native encoder samples, never the upsampled reference returned
    # by the legacy preparation routine. No claim that source policy actions
    # were applied, or that the body root was observed.
    q,t=a["source_state43"],a["source_timestamps"]
    names=np.r_[a["body_joint_names"],a["hand_joint_names"]].tolist()
    pts,stream=video_pts(video)
    if len(pts)!=len(t) or not np.allclose(pts,t,rtol=0,atol=2e-5):
        raise ValueError("Native robot encoder samples do not match the ego RGB clock")
    n=min(len(t),frames)
    width,height=stream["width"],stream["height"]
    raw=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-threads","1","-i",str(video),
        "-frames:v",str(n),"-pix_fmt","rgb24","-threads","1","-f","rawvideo","pipe:1"])
    images=np.frombuffer(raw,dtype=np.uint8).reshape(n,height,width,3)
    return c,source,parquet,video,mapping,names,q,t,pts,images


def run(ep,out,frames,roi,*, small_path=None, sam3_path=None, task_path=None, depth_path=None, device="cuda:0"):
    cfg=config()
    task=read_task(task_path,cfg) if task_path else None
    tracker="sam3" if sam3_path else "red_component_tracker"
    if task and task["tracker"]!=tracker:
        raise ValueError("Loaded perception backend differs from the fixed task setup")
    c,source,parquet,video,mapping,names,q,t,pts,images=load_capture(cfg,ep,frames)
    n=len(images)
    fk=FixedBaseFK(compact_model(c/"scene/scene.xml"),names)
    track=Sam3Mask(sam3_path,"apple",device,roi) if sam3_path else AppleMask(roi)
    joints=Latest(.06)
    perception=Latest(.15)
    phase=Phase(cfg["phase"])
    priors=task["priors"] if task else prepare("apple",ROOT/"configs/object_sources.json")
    if depth_path:
        if priors["dimensions"] is None:
            raise ValueError("RGB-D center estimation needs an independent metric diameter in the task priors")
        meta,depth=recorded_depth(depth_path,video,pts,images.shape[1:3],domain="recorded_hardware")
        optical=DepthObservations(meta["intrinsics"],meta["base_from_optical"],max(priors["dimensions"]["value"]))
        markers={}
        required=set(depth["rgb_index"][depth["rgb_index"]<n])
        d=0
    result=[]
    first=True
    if small_path:
        import torch
        from .models import Q, gather, load_models
        from .pretrain import ensemble
        from .rollout import policy_load
        policy=policy_load(cfg,device)
        actor,dynamics,small=load_models(small_path,device)
        if names!=list(small["joint_names"]) or len(small.get("critics",[]))!=2:
            raise ValueError("Replay requires both compatible SAC critics in the supplied checkpoint")
        critics=[Q().to(device).eval() for _ in small["critics"]]
        for model,state in zip(critics,small["critics"]):
            model.load_state_dict(state)
        root=np.r_[fk.origin,wxyz(fk.axes.inv())]
        # Camera/joint samples here are real archived inputs. No previous
        # applied commands, 50Hz transitions or rewards exist in this archive.
        plan=None
        plans,actor_frames,dynamics_frames=[],[],[]
        time=np.maximum(t[:n],pts[:n])
        phys=physical(priors)
    for i,(tq,trgb,rgb) in enumerate(zip(t[:n],pts[:n],images)):
        joints.put(tq,dict(q=q[i]))
        marker=track(rgb,trgb)
        perception.put(trgb,marker)
        now=max(tq,trgb)
        pose=None
        if depth_path:
            if i in required:
                markers[i]=marker
            while d<len(depth["available_s"]) and depth["available_s"][d]<=now:
                j=int(depth["rgb_index"][d])
                optical.publish(markers.pop(j),depth["depth_m"][d],depth["captured_s"][d],depth["available_s"][d])
                d+=1
            pose=optical.read(now)
        # A center estimate supplies geometry, not rotation, contact, or a success witness.
        joint=joints.read(now)
        ob,cloud=observation(joint,pose,fk,priors,now)
        row=phase.advance(now,ob,False,stable_interval=None)
        result.append(dict(time=now,dynamics=dyn_frame(ob,None),mask_pixels=int(marker["mask"].sum()),
            mask_confidence=marker["confidence"],phase=row[0],phase_valid=row[1],point_valid=cloud["point_valid"].sum(),
            max_source_age=max(now-min(tq,trgb),pose[2] if pose else 0.),wrist=ob["wrist"]))
        if depth_path:
            result[-1].update(object_center_m=ob["object"][:3],object_center_valid=ob["object_valid"],
                object_center_capture_s=pose[1] if pose else np.nan,object_center_confidence=ob["object_confidence"],
                observed_points=cloud["points"],observed_point_valid=cloud["point_valid"],observed_point_conf=cloud["point_conf"])
        if small_path:
            tick=int(np.floor((now-time[0]+1e-6)/.02))
            start=tick-tick%20
            if not plan or int(plan[0]["first_physical_frame"])-1!=start:
                plan=nominal_from_capture(policy,fk,joint,(rgb,trgb),root,start,now,cfg["task"],cfg["seed"],
                                          plan[0] if plan else None)
                plans.append(plan[0])
            offset=tick-start
            af,df=hardware_features(ob,plan[1][offset:],plan[2][offset:],None,None,row,
                                    max(0.,len(t)/30.-now),offset*.02)
            actor_frames.append(af)
            dynamics_frames.append(df)
            h,hn=window(np.asarray(actor_frames),time,i,cfg["history"])
            d,dn=window(np.asarray(dynamics_frames),time,i,cfg["history"])
            batch=gather([dict(history=h,length=hn,physical=phys,**cloud)],device)
            with torch.inference_mode():
                z,_=actor.sample(batch,deterministic=True)
                values=torch.stack([qnet(batch,z) for qnet in critics])[:,0]
                # This assesses the predictor input contract for a nominal
                # candidate, never a target/label or a verified transition.
                predicted,_,_,event=ensemble(dynamics,gather([dict(history=d,length=dn,physical=phys,
                    action=plan[1][offset],dt=[.02],**cloud)],device))
            v=[x.cpu().numpy() for x in (z[0],values,predicted[0],event[0])]
            masked=np.r_[43:86,89:95] if pose else np.arange(43,101)
            if (not all(np.isfinite(x).all() for x in v) or af[1,masked].any() or df[1,masked].any()
                    or df[1,101:144].any()):
                raise ValueError("Deployment planner used unavailable channels or produced nonfinite outputs")
            result[-1].update(actor_history_frame=af, dynamics=df, nominal_43=plan[1][offset],
                              prospective_residual=v[0],q_values=v[1],d0_candidate_delta=v[2],d0_candidate_events=v[3])
        if first and marker["centroid"] is not None:
            annotated=rgb.copy()
            annotated[marker["mask"]]=((.5*annotated[marker["mask"]])+[0,112,0]).clip(0,255).astype(np.uint8)
            Image.fromarray(annotated).save(out/"tracked_rgb.png")
            first=False
    if (joints.read(ts:=t[n-1]+.07) is not None or perception.read(pts[n-1]-.001) is not None
            or phase.outcome(True)["success"] or any(r["phase_valid"] for r in result)
            or not depth_path and any(r["point_valid"] for r in result)):
        raise ValueError("Stale or absent evidence was interpreted as a present packet or success")
    save_arrays(out/"observations.npz",**{k:np.asarray([r[k] for r in result]) for k in result[0]},
                q=q[:n],joint_names=np.array(names),native_time_s=t[:n],rgb_pts_s=pts[:n],
                physical=physical(priors))
    write(out/"priors.json",priors)
    policy_check=None
    if small_path:
        keys=("plan_id","first_physical_frame","prefix_length","measured_body","measured_hand","anchor",
              "body_raw","hand_absolute","total_inference_seconds")
        save_arrays(out/"plans.npz",**{k:np.asarray([p[k] for p in plans]) for k in keys})
        policy_check=dict(small_sha256=sha256(small_path),parent=policy.loading_receipt,plans=len(plans),
                          actor_critic_and_d0_frames=n,max_planner_inference_s=max(p["total_inference_seconds"] for p in plans),
                          replay_has_no_applied_commands=True,not_a_realtime_measurement=True)
    write(out/"summary.json",dict(source="original AppleToPlate robot encoders and ego RGB",
        source_revision=source["revision"],episode=ep,frames=n,recording_fps=30,
        provenance={str(p):sha256(p) for p in (parquet,video)},joint_order=mapping,
        rgb_state_pts_error_max_s=float(np.max(abs(pts-t))),
        color_track_detections=None if sam3_path else sum(r["mask_pixels"]>0 for r in result),
        tracker=marker["source"],tracked_mask_frames=sum(r["mask_pixels"]>0 for r in result),offline_policy_check=policy_check,
        task_setup_sha256=sha256(task_path) if task_path else None,
        detections_are_unlabelled=True,encoder_stale_rejected_at_s=ts,
        registered_depth_sha256=sha256(depth_path) if depth_path else None,
        metric_geometry_frames=sum(bool(r["point_valid"]) for r in result),
        metric_center_estimates=sum(bool(r.get("object_center_valid")) for r in result),
        known_pose_frames=0,success_evidence_frames=0,closed_loop=False,hardware_command_sent=False,
        absent=(["synchronized depth and camera-to-base calibration"] if not depth_path else [])+
               ["verified full object pose","joint velocity",
                "timestamped applied commands","physical contact and success witness"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--frames",type=int,default=90)
    p.add_argument("--roi",nargs=4,type=int)
    p.add_argument("--small",type=Path,help="Also run frozen Movens, compatible actor, both Qs and D0 offline")
    p.add_argument("--sam3-dir",type=Path,help="Local SAM 3 weights in a compatible perception environment")
    p.add_argument("--task-info",type=Path,help="Precomputed fixed task and tool configuration")
    p.add_argument("--depth-manifest",type=Path,help="Registered metric depth and availability stamps on the RGB clock")
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    if a.frames<2:
        p.error("At least two native frames are required")
    a.out.mkdir(parents=True,exist_ok=True)
    run(a.episode,a.out,a.frames,a.roi,small_path=a.small,sam3_path=a.sam3_dir,task_path=a.task_info,
        depth_path=a.depth_manifest,device=a.device)


if __name__=="__main__":
    main()

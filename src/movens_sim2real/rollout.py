"""Frozen Movens + bounded hand residual; all D0 predictions are logged off-policy."""
import argparse
from contextlib import closing as close_render
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

import mujoco as mj
import numpy as np
import torch
from PIL import Image

from movens_interact.model import MovensInteractModel
from simulation.stage4_videos import PREFIX_KEYS, SceneGuard, Views

from .contract import LEFT, ROOT, SCHEMA, candidate, config, read, save_arrays, sha256, write
from .data import closure, package, split_apple
from .features import actor_frame, dyn_frame, packed, point_context, window
from .hardware import hardware_features, hardware_phase
from .models import gather, load_models, save_checkpoint
from .phase import CARRY, PLACE, Phase, RULE_VERSION, phase_arrays
from .pretrain import ensemble
from .scene import NativeScene


PLAN_ARRAYS = ("body_raw","body_normalized","body_absolute","hand_normalized","hand_absolute",
               "hand_absolute_raw","hand_execution_normalized","measured_body","measured_hand",
               "body_state_normalized","hand_state_normalized","anchor","first_physical_frame",
               "prefix_length","plan_id","context_seconds","body_seconds","hand_seconds","total_inference_seconds",
               "prefix_body_max_abs_error","prefix_hand_max_abs_error")


def file_identity(p):
    st=Path(p).stat()
    return dict(path=str(Path(p).resolve()), size=st.st_size,mtime_ns=st.st_mtime_ns)


def policy_load(cfg, device, replacement=None):
    checkpoint=replacement or cfg["checkpoint"]
    run=read(cfg["baseline_eval"]/"run_config.json")
    runtime=torch.device(device)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    if runtime.type=="cuda":
        torch.cuda.set_device(runtime)
        torch.cuda.set_per_process_memory_fraction(.9,runtime)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    model=MovensInteractModel.from_pretrained(checkpoint,base_checkpoint=run["base_checkpoint"],
                                                backbone=run["qwen_config"],device=device)
    for k in ("optimizer_state_dict","rng_state","body_frozen_probe_reference","runtime_state"):
        model.checkpoint.pop(k,None)
    if model.prompt!=cfg["task"] or not model.loading_receipt["frozen"]:
        raise ValueError("Unexpected frozen policy prompt or checkpoint")
    print(f"Frozen policy {model.loading_receipt}",flush=True)
    return model


def reward_progress(obs):
    # Bounded task-relative progress. No point is given for moving the
    # apple alone: after approaching, lift/carry bonuses need physical events.
    near=1-min(max(np.min(obs["tips"][:,0]),0.)/.22,1.)
    return .5*near


def build_video(out, view):
    dest=out/(view+".mp4")
    partial=dest.with_name(view+".partial.mp4")
    src=out/("rgb" if view=="ego" else view)
    if not any(src.glob("*.png")):
        return None
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-threads","1","-framerate","10",
                    "-pattern_type","glob","-i",str(src/"*.png"),"-c:v","libx264","-threads","2",
                    "-pix_fmt","yuv420p","-movflags","+faststart",str(partial)],check=True)
    partial.replace(dest)
    return dest


def record_image(path, img):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".partial.png")
    Image.fromarray(img).save(temp,compress_level=1)
    temp.replace(path)


def settle_original(scene, guard, adapter=None):
    stamp=scene.restore_original(-scene.settle)
    if adapter:
        stamp=adapter.after_reset()
    n=round(scene.settle/scene.dt)
    for _ in range(n):
        mj.mj_step(scene.m,scene.d)
        err=guard.check(float(scene.d.time))
        if err:
            raise FloatingPointError(f"Invalid original settle: {err}")
    scene.check(-scene.settle,n)
    return stamp,scene.observe()


def _rng_key(seed,episode):
    return int(np.random.SeedSequence([seed,episode]).generate_state(1)[0])


def episode_rollout(cfg,ep,out,policy,actor=None,dynamics=(),*,small_identity=None,seed=42,trial=0,
                    train=False,side=False,seconds=None,decision_limit=None,checkpoint_every=90,
                    adapter_factory=None,phase_mode="phase",history_length=None,stop=None):
    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    device=policy.device
    dt=cfg["control_dt"]
    c=candidate(cfg,ep)
    split=split_apple(cfg)[ep]
    if train and split!="train":
        raise ValueError("Learning cannot collect from a held-out source group")
    scene=NativeScene(c,cfg)
    adapter=adapter_factory(scene,_rng_key(seed,ep+100000*trial)) if adapter_factory else None
    identity=dict(schema=SCHEMA,rule_version=RULE_VERSION,rules=cfg["phase"],episode=ep,trial=trial,
                  source_group_id=f"apple-d89c126a-{ep:06d}",reset_id=trial,split=split,
                  dt=dt,physics_dt=scene.dt,seed=seed,task=cfg["task"],training=train,
                  mode="D0",residual_enabled=actor is not None,d0_predictions=bool(dynamics),side=side,
                  phase_mode=phase_mode,history_length=history_length or cfg["history"],
                  adapter=adapter.identity if adapter else None,asset=scene.assets(),
                  code={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name("scene.py"),
                                                Path(__file__).with_name("features.py"),Path(__file__).with_name("phase.py"))},
                  policy=file_identity(policy.adapter_checkpoint),small=small_identity,
                  seconds=seconds,decision_limit=decision_limit)
    if adapter:
        identity["code"].update({n:sha256(Path(__file__).with_name(n)) for n in ("deployment.py","hardware.py","vision.py")})
    cp=out/"resume.pt"
    summary=out/"summary.json"
    request=out/"run_config.json"
    import json
    if request.exists():
        # Use the JSON writer's canonical conversion for comparison across restarts.
        from .contract import _default
        if read(request)!=json.loads(json.dumps(identity,default=_default)):
            raise ValueError(f"{out}: rollout configuration changed")
    else:
        write(request,identity)
    identity=read(request)
    if summary.exists():
        return read(summary)
    total_task=round(scene.end/dt)
    full=total_task+round(scene.tail/dt)
    horizon=min(full,round(seconds/dt) if seconds is not None else full,
                decision_limit if decision_limit is not None else full)
    if not 0<horizon<=full:
        raise ValueError("No physical step budget")
    generator=torch.Generator(device=device).manual_seed(_rng_key(seed,ep+100000*trial))
    guard=SceneGuard(scene.sim,scene.report["identity"]["policy"])
    if cp.exists():
        z=torch.load(cp,map_location="cpu",weights_only=False)
        if z["identity"]!=identity:
            raise ValueError("Incompatible rollout checkpoint")
        scene.restore(z["integration"],resume=True)
        guard.state,guard.contact6_seen=z["guard"]
        phase,observed_phase=z["phase"],z["observed_phase"]
        true,observed=z["true"],z["observed"]
        previous,correction,previous_nominal=z["previous"],z["correction"],z["previous_nominal"]
        true_stable=z["true_stable"]
        a_frames,d_frames,observations,commands,plans,phases,vis_context,frames=(
            z[k] for k in ("a_frames","d_frames","observations","commands","plans","phases","vis_context","frames"))
        generator.set_state(z["rng"])
        i=z["tick"]
        initial_target=z["initial_target"]
        first_events=z["first_events"]
        if adapter:
            adapter.load_state(z["adapter"])
    else:
        orig,true=settle_original(scene,guard,adapter)
        save_arrays(out/"initial.npz", reset=orig,integration=scene.snapshot(),
                    qpos=scene.d.qpos,qvel=scene.d.qvel,actuator=scene.d.ctrl)
        i=0
        initial_target=scene.d.ctrl[scene.sim.actuator_ids].copy()
        previous,previous_nominal=initial_target.copy(),initial_target.copy()
        correction=np.zeros(7)
        true_stable=False
        observed=None
        phase=Phase(cfg["phase"])
        observed_phase=Phase(cfg["phase"]) if adapter else None
        a_frames,d_frames,observations,commands,plans,phases,vis_context,frames=([] for _ in range(8))
        first_events=[False,False]
    view_entry=dict(candidate=c,scene_xml=str(scene.sim.scene_path))
    wall=time.monotonic()
    err=None
    plan=plans[-1] if plans else None
    wrist_plan=scene.sensors.wrist_plan(plan["body_absolute"][:,:29]) if plan else None
    capacity=identity["history_length"]
    tt=np.arange(horizon+1)*dt
    exact_points=point_context(scene.sensors.point_cloud)
    exact_phys=packed(scene.sensors.physical,scene.sensors.physical_valid)
    with close_render(Views(view_entry)) as viewer:
        if adapter:
            adapter.prepare_view(viewer)
        while i <= horizon:
            t=i*dt
            active=i<total_task
            refresh=active and i%20==0
            rgb=None
            if refresh or i%5==0:
                viewer.synchronize(scene.d.qpos,scene.d.qvel,scene.d.ctrl)
                rgb=viewer.image("ego")
                if i%5==0 and (not frames or frames[-1]<i):
                    record_image(out/"rgb"/f"{i:06d}.png",rgb)
                    if side:
                        record_image(out/"side"/f"{i:06d}.png",viewer.image("side"))
                    frames.append(i)
            if adapter:
                observed=adapter.observe(t,true,rgb)
            if refresh:
                body,hand=scene.sim.observe()
                if adapter:
                    body[0,:29]=observed["q"][:29]
                    hand[0]=observed["q"][29:]
                prev={k:plan[k] for k in PREFIX_KEYS} if plan else None
                before=float(scene.d.time)
                p=policy.predict(rgb,body,hand,i,previous_plan=prev,seed=seed,task_text=cfg["task"])
                if float(scene.d.time)!=before or p["first_physical_frame"]!=i+1 or p["prefix_length"]!=(0 if i==0 else 10):
                    raise ValueError("Original action block/control timing changed")
                plan={k:p[k] for k in PLAN_ARRAYS}
                plans.append(plan)
                wrist_plan=scene.sensors.wrist_plan(plan["body_absolute"][:,:29])
            offset=i%20 if active else (total_task-1)%20
            targets=np.c_[plan["body_absolute"][offset:,:29],plan["hand_absolute"][offset:]]
            wrist=wrist_plan[offset:]
            nominal=targets[0].astype(np.float64) if active else previous_nominal
            closing=bool(closure(nominal,previous_nominal,dt,cfg["phase"]))
            true_phase=phase.advance(t,true,closing,stable_interval=true_stable)
            phases.append(true_phase)
            if adapter:
                ob=observed
                measured="orientation_valid" in ob
                control_phase=(hardware_phase(observed_phase,t,ob,closing) if measured else
                               observed_phase.advance(t,ob,closing,stable_interval=None))
                point,phys=adapter.context()
            else:
                ob={**true,"age_s":0.,"time":t}
                control_phase=true_phase
                point,phys=exact_points,exact_phys
                measured=False
            task_left,chunk_age=max(0.,scene.end-t),offset*dt
            if measured:
                af,df=hardware_features(ob,targets,wrist,previous,correction,control_phase,
                                        task_left,chunk_age,ablation=phase_mode)
            else:
                af=actor_frame(ob,targets,wrist,previous,correction,control_phase,task_left,chunk_age,ablation=phase_mode)
                df=dyn_frame(ob,previous)
            a_frames.append(af)
            d_frames.append(df)
            observations.append(dict(true,stable_interval=bool(true_stable)))
            obs_meta=np.array([ob.get("object_valid",True),ob.get("object_confidence",1.),ob.get("age_s",0.),
                               ob.get("support_valid",True),control_phase[0],control_phase[1],control_phase[2]],dtype=np.float32)
            vis_context.append(dict(**point,physical=phys,metadata=obs_meta,estimate_object=ob["object"],
                                    estimate_twist=ob["twist"],estimate_q=ob["q"],estimate_qd=ob["qd"],
                                    estimate_wrist=ob["wrist"],estimate_target=ob["target"],
                                    estimate_q_capture_s=ob.get("q_time_s",t-ob.get("age_s",0.)),
                                    estimate_object_capture_s=(ob.get("object_time_s",t-ob.get("age_s",0.))
                                                               if ob.get("object_valid",True) else np.nan),
                                    estimate_groups=ob["groups"],estimate_groups_valid=ob["groups_valid"]))
            if commands:
                new_grasp=phase.ever_grasped and not first_events[0]
                new_goal=phase.goal_now and not first_events[1]
                first_events[0]|=new_grasp
                first_events[1]|=new_goal
                end=i==horizon and horizon==full
                accepted=end and phase.outcome(True)["success"]
                zv=commands[-1]["residual_proposal"]
                zl=commands[-2]["residual_proposal"] if len(commands)>1 else np.zeros(7)
                previous_obs=observations[-2]
                aux=.2*(.99*(0. if end else reward_progress(true))-reward_progress(previous_obs))
                reward_vec=np.array([10.*accepted,1.*new_grasp,1.*new_goal,-5.*true_phase[-1],aux,
                                     -.001*np.mean(zv*zv),-.0005*np.mean((zv-zl)**2)],dtype=np.float32)
                commands[-1].update(reward_terms=reward_vec,reward=float(reward_vec.sum()),
                                     terminated=end,truncated=i==horizon and horizon<full)
            if i==horizon:
                break
            proposal=np.zeros(7)
            if active and actor is not None:
                h,n=window(np.asarray(a_frames[-capacity:]),tt[max(0,i+1-capacity):i+1],min(i,capacity-1),capacity)
                with torch.inference_mode():
                    small=gather([dict(history=h,length=n,**point)],device)
                    proposal=actor.sample(small,not train,generator)[0][0].cpu().numpy().astype(np.float64)
            if active:
                target,correction=scene.limits.execute(nominal,proposal,correction)
                if actor is None and not np.array_equal(target,np.clip(nominal,scene.limits.lo,scene.limits.hi)):
                    raise ValueError("Zero residual must equal original controller limits")
            else:
                target=previous.copy()  # Original release verification holds final command.
            prediction={}
            if dynamics:
                h,n=window(np.asarray(d_frames[-capacity:]),tt[max(0,i+1-capacity):i+1],min(i,capacity-1),capacity)
                b=gather([dict(history=h,length=n,physical=phys,action=target,dt=[dt],**point)],device)
                with torch.inference_mode():
                    pred,pvar,disag,event=ensemble(dynamics,b)
                    prediction=dict(predicted_delta=pred[0].cpu().numpy(),predicted_variance=pvar[0].cpu().numpy(),
                                    predicted_disagreement=disag[0].cpu().numpy(),predicted_events=event[0].cpu().numpy())
            try:
                close=phase.stage in (CARRY,PLACE) and true["margin"]>=-.03
                contact,true_stable=scene.step(target,observe_stability=close,guard=guard)
            except FloatingPointError as e:
                err=str(e)
                save_arrays(out/"failure_state.npz",integration=scene.snapshot(),attempted_target=target)
                break
            actual=scene.d.ctrl.copy()
            if not np.array_equal(actual[scene.sim.actuator_ids],target):
                raise ValueError("Actual low-level actuator target mismatch")
            true=scene.observe(contact)
            commands.append(dict(nominal=nominal,residual_proposal=proposal,selected_residual=proposal,
                                 executed=target,executed_correction=correction.copy(),actuator=actual,
                                 decision_mask=active,plan_id=int(plan["plan_id"]),plan_offset=offset,
                                 tick=i,t_command=t,t_command_end=(i+1)*dt,**prediction))
            previous,previous_nominal=target,nominal
            i+=1
            interrupted=stop is not None and stop()
            if i%checkpoint_every==0 or interrupted:
                save_checkpoint(cp,dict(identity=identity,integration=scene.snapshot(),guard=(guard.state,guard.contact6_seen),
                    phase=phase,observed_phase=observed_phase,true=true,observed=observed,previous=previous,
                    correction=correction,previous_nominal=previous_nominal,true_stable=true_stable,
                    a_frames=a_frames,d_frames=d_frames,observations=observations,commands=commands,plans=plans,phases=phases,
                    vis_context=vis_context,frames=frames,rng=generator.get_state(),tick=i,initial_target=initial_target,
                    first_events=first_events,adapter=adapter.state() if adapter else None))
                print(f"episode={ep:03d} tick={i}/{horizon} elapsed={time.monotonic()-wall:.1f}s",flush=True)
            if interrupted:
                return None
    complete=err is None and i==full
    result=phase.outcome(complete,engineering_error=err)
    n_commands=len(commands)
    if n_commands!=len(observations)-1 or (
        n_commands and any(k not in commands[-1] for k in ("reward","terminated","truncated"))):
        raise ValueError("Decision log incomplete at finalization")
    # Correlate each pre-action state with its actual following interval.
    # A failed unfinished physical command is stored separately for diagnosis.
    s=scene.sensors
    a=package(observations,s,tt[:len(observations)],np.zeros((0,43)),
              np.asarray([x["stable_interval"] for x in observations]),cfg,c,scene.limits,scene.sim.actuator_ids)
    if n_commands:
        a.update({k:np.asarray([p[k] for p in commands]) for k in commands[0]})
        a["all_commands"]=np.vstack((a["executed"],a["executed"][-1:]))
    a.update(phase_arrays(phases))
    a.update({f"observed_{k}":np.asarray([p[k] for p in vis_context]) for k in vis_context[0]})
    a.update(actor_frames=np.asarray(a_frames),dynamics_frames=np.asarray(d_frames),
             rgb_ticks=np.asarray(frames,dtype=np.int64),rgb_time_s=dt*np.asarray(frames,dtype=np.float64),
             motion_end_s=scene.end,initial_target=initial_target,continuous_reset=np.array(True),
             events=np.c_[a["groups"],a["slip"],a["target_support"]],
             events_valid=np.c_[a["groups_valid"],a["phase_valid"],np.ones(len(a["t"]),bool)],
             actual_20ms=np.ones(n_commands,bool),control_ticks=np.arange(len(a["t"])))
    save_arrays(out/"trajectory.npz",**a)
    save_arrays(out/"plans.npz",**{k:np.asarray([p[k] for p in plans]) for k in PLAN_ARRAYS})
    save_arrays(out/"terminal.npz",integration=scene.snapshot())
    np_any=np.asarray(a["robot_contact"],bool).any()
    result.update(identity=identity,path=out/"trajectory.npz",engineering_error=err,complete=complete,
                  decision_steps=n_commands,active_steps=int(np.count_nonzero(a.get("decision_mask",[]))),
                  grasp_contact_any=bool(np_any),frozen_model=policy.loading_receipt,
                  policy_calls=len(plans),interaction_s=n_commands*dt,wall_seconds=time.monotonic()-wall,
                  rgb_frames=len(frames),rgb_video=build_video(out,"ego"),side_video=build_video(out,"side") if side else None,
                  reached_protocol_initial=True,success_export_eligible=bool(result["success"] and split=="train"),
                  record_streams=["target","actual",*(["dynamics_prediction"] if dynamics else [])])
    write(summary,result)
    if cp.exists():
        cp.unlink()
    print(f"episode={ep:03d} outcome={result['failure'] or 'success'} steps={n_commands}",flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/experiment.json")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--episode",type=int,nargs="+",required=True)
    p.add_argument("--small",type=Path)
    p.add_argument("--base-adapter",type=Path)
    p.add_argument("--zero",action="store_true")
    p.add_argument("--no-dynamics",action="store_true")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--seconds",type=float)
    p.add_argument("--checkpoint-every",type=int,default=90)
    p.add_argument("--side",action="store_true")
    args=p.parse_args()
    cfg=config(args.config)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".run.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        requested=[]
        for signum in (signal.SIGTERM,signal.SIGINT):
            signal.signal(signum,lambda *_:requested.append(True))
        policy=policy_load(cfg,args.device,args.base_adapter)
        actor,dynamics,small=None,[],None
        if args.small:
            actor,dynamics,_=load_models(args.small,args.device)
            small=dict(path=args.small.resolve(),sha256=sha256(args.small))
        if args.no_dynamics:
            dynamics=[]
        for ep in args.episode:
            if requested:
                break
            outcome=episode_rollout(cfg,ep,args.out/f"apple_{ep:06d}",policy,
                                    None if args.zero else actor,dynamics,small_identity=small,
                                    seed=args.seed,side=args.side,seconds=args.seconds,
                                    checkpoint_every=args.checkpoint_every,stop=lambda:bool(requested))
            if outcome is None:
                break


if __name__=="__main__":
    main()

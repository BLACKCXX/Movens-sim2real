"""Run a bounded MuJoCo integration check of deployment observations."""
import argparse
from pathlib import Path
import signal

import numpy as np

from movens_sim2real.contract import config, sha256, write
from movens_sim2real.data import array_dict
from movens_sim2real.deployment import factory
from movens_sim2real.export import audit
from movens_sim2real.models import load_models
from movens_sim2real.rollout import episode_rollout, policy_load


def check_resume(out, reference):
    differences,counts=[],{}
    for name in ("initial.npz","trajectory.npz","plans.npz","terminal.npz"):
        a,b=(array_dict(p/name) for p in (reference,out))
        if a.keys()!=b.keys():
            raise ValueError(f"Restore changed the schema of {name}")
        keys=[k for k in a if not k.endswith("_seconds")]
        counts[name]=len(keys)
        differences.extend(f"{name}:{k}" for k in keys if not np.array_equal(a[k],b[k],equal_nan=a[k].dtype.kind in "fc"))
    images=sorted((reference/"rgb").glob("*.png"))
    differences.extend(f"rgb/{p.name}" for p in images if sha256(p)!=sha256(out/"rgb"/p.name))
    write(out/"equivalence.json",dict(reference=str(reference),restored=str(out),fields_compared=counts,
          rgb_frames=len(images),differences=differences,exact=not differences))
    if differences:
        raise ValueError(f"Resumed native physics or deployment history disagrees: {differences}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--episode",type=int,default=1)
    p.add_argument("--trial",type=int,default=0,help="Use the same fixed disturbance as this S1 collection slot")
    p.add_argument("--seconds",type=float,default=1.8)
    p.add_argument("--pause-after",type=int,help="Checkpoint after this many steps of a new probe")
    p.add_argument("--reference",type=Path,help="Compare a completed resumed probe against an uninterrupted rollout")
    a=p.parse_args()
    if a.pause_after is not None and (not 0<a.pause_after<round(a.seconds/.02) or (a.out/"run_config.json").exists()):
        p.error("A planned pause must be inside a new probe")
    cfg=config()
    model=policy_load(cfg,"cuda:0")
    actor,dynamics,_=load_models(a.small,"cuda:0")
    stop=[]
    for s in (signal.SIGTERM,signal.SIGINT):
        signal.signal(s,lambda *_:stop.append(True))
    decisions=0
    def paused():
        nonlocal decisions
        decisions+=1
        return bool(stop) or decisions==a.pause_after
    small=dict(path=a.small.resolve(),sha256=sha256(a.small))
    summary=episode_rollout(cfg,a.episode,a.out,model,actor,dynamics,small_identity=small,
        seed=42,trial=a.trial,seconds=a.seconds,adapter_factory=factory("deployment"),checkpoint_every=30,
        stop=paused)
    if summary is None:
        return
    _,z,_=audit(a.out/"summary.json")
    est=z["observed_metadata"]
    known=est[:,0].astype(bool)
    mask=z["observed_point_valid"].sum(axis=1)
    phys=z["observed_physical"]
    af,df=z["actor_frames"],z["dynamics_frames"]
    qd=af[:,1,43:86].all(axis=1)
    timestamps=z["t"]
    if (not np.any(~known) or not np.any(known)
            or np.any(mask[~known]) or not np.any(mask[known])
            or np.any(phys[:,1,:16]) or np.any(phys[:,1,26:])
            or np.any(est[:,2]<0) or not np.isfinite(z["predicted_delta"]).all()
            or not np.any(qd) or not np.any(~qd)
            or not np.array_equal(df[:,1,43:86].all(axis=1),qd)
            or np.any(af[:,1,89:95]) or np.any(df[:,1,89:95])
            or np.any(af[:,1,98:101]) or np.any(df[:,1,98:101])
            or np.any(z["observed_estimate_q_capture_s"]>timestamps+1e-9)
            or np.any(z["observed_estimate_object_capture_s"][known]>timestamps[known]+1e-9)
            or not np.array_equal(np.isfinite(z["observed_estimate_object_capture_s"]),known)):
        raise ValueError("No evidence for both visibility and masked deployment contexts, or invalid D0 output")
    from movens_sim2real.learn import EpisodeTransitions, SAC
    import torch
    # Exercise both learned Qs and the predictor on actual observed replay
    # rows. This integration check never changes the saved deployment policy.
    e=EpisodeTransitions(summary,25,.99)
    opt=argparse.Namespace(device="cuda:0",actor_lr=1e-4,dynamics_lr=1e-4,lr=3e-4)
    agent=SAC(actor,dynamics,opt)
    from movens_sim2real.models import gather
    ids=np.array([0,min(20,len(z["executed"])-1),len(z["executed"])-1])
    rb=gather([e.rl(int(i)) for i in ids],"cuda:0")
    db=gather([e.dynamic(int(i)) for i in ids],"cuda:0")
    loss=agent.update(rb,[db for _ in dynamics])
    write(a.out/"verification.json",dict(physical_profile=summary["identity"]["adapter"],
          actual_20ms_steps=len(z["executed"]),visible_frames=int(known.sum()),missing_frames=int((~known).sum()),
          missing_encoder_velocity_frames=int((~qd).sum()), orientation_frames=int(af[:,1,89:95].all(axis=1).sum()),
          point_counts=[int(mask[known].min()),int(mask[known].max())],
          max_observation_age_s=float(est[:,2].max()),
          max_encoder_age_s=float(np.max(timestamps-z["observed_estimate_q_capture_s"])),
          max_measured_position_error_m=float(np.linalg.norm(
              z["observed_estimate_object"][known,:3]-z["object"][known,:3],axis=1).max()),
          real_logged_transition_update=loss,acceptance=summary["failure"],
          note="S1 observation integration check using the supplied checkpoint; not formal S1 training or evaluation"))
    if a.reference:
        check_resume(a.out,a.reference)


if __name__=="__main__":
    main()

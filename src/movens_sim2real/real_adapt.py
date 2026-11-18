"""Adapt SAC heads between measured hardware trial batches; never command a robot."""
import argparse
import fcntl
from pathlib import Path
import signal

import numpy as np
import torch

from .calibration import Trial
from .contract import config, read, sha256, write
from .data import closure
from .features import ACTOR_SIZE, window
from .hardware import hardware_phase
from .learn import SAC
from .models import gather, load_models, save_checkpoint
from .phase import Phase
from .pretrain import restore_rng, rng_state


CUES=("object","twist","wrist","wrist_twist","tips","diameter","margin","groups","groups_valid",
      "original_support","target_support","robot_contact","object_valid","wrist_valid","wrist_twist_valid",
      "linear_valid","angular_valid","diameter_valid","tips_valid","margin_valid","support_valid",
      "robot_contact_valid","confidence")


def measured_returns(tr, cfg, *, require_replay=True):
    """Use only measured events; unknown intervals never become success or negative terminal labels."""
    a,m=tr.a,tr.meta
    n=len(a["t"])
    active=np.flatnonzero(a["decision_mask"])
    z=a["selected_residual"]
    if (not len(active) or not np.array_equal(active,np.arange(len(active))) or active[-1]>=n-1
            or a["decision_mask"].dtype!=bool or a["decision_mask"].shape!=(n-1,)
            or z.shape!=(n-1,7) or not np.isfinite(z).all() or np.any(abs(z)>1+1e-6)
            or a["actor_frames"].shape!=(n,3,ACTOR_SIZE) or not np.isfinite(a["actor_frames"]).all()
            or a["nominal"].shape!=(n,43) or not np.isfinite(a["nominal"]).all()
            or any(a[k].shape!=(n,) or a[k].dtype!=bool for k in ("stable_interval","stable_interval_valid"))
            or any(len(a[k])!=n for k in CUES)):
        raise ValueError(f"{tr.path}: missing synchronized deployed policy or measured phase evidence")
    if m["task"]!=cfg["task"] or m["phase_rules"]!=cfg["phase"] or not m["success_evidence_source"]:
        raise ValueError(f"{tr.path}: changed task, thresholds or success evidence")
    phase=Phase(cfg["phase"])
    reward=np.zeros(n-1,dtype=np.float32)
    trusted=np.zeros(n,bool)
    grasp_seen=goal_seen=False
    for i,t in enumerate(a["t"]-a["t"][0]):
        ob={k:a[k][i] for k in CUES}
        if not .6<=ob["confidence"]<=1.:
            ob.update(object_valid=False,groups_valid=np.zeros(3,bool),support_valid=False,robot_contact_valid=False)
        closing=bool(closure(a["nominal"][i],a["nominal"][max(i-1,0)],.02,cfg["phase"]))
        stable=(True if i and trusted[i-1] and a["stable_interval_valid"][i] and a["stable_interval"][i]
                and all(ob[k] for k in ("object_valid","linear_valid","angular_valid","margin_valid",
                                        "support_valid","target_support","robot_contact_valid"))
                and ob["margin"]>=0 and not ob["robot_contact"] and not ob["original_support"] else None)
        row=hardware_phase(phase,t,ob,closing,stable_interval=stable)
        trusted[i]=row[1] and row[2]>=.6
        if i and trusted[i]:
            reward[i-1]=float(phase.ever_grasped and not grasp_seen)+float(phase.goal_now and not goal_seen)-5.*row[-1]
        grasp_seen|=phase.ever_grasped
        goal_seen|=phase.goal_now
    outcome=phase.outcome(m["complete"],continuous=True,engineering_error=m.get("engineering_error"))
    if m.get("engineering_error"):
        raise ValueError(f"{tr.path}: engineering-invalid attempt cannot update SAC")
    outcome_known=outcome["success"] or (m["complete"] and trusted.all() and a["stable_interval_valid"][1:].all())
    if outcome["success"]:
        reward[-1]+=10.
    # Keep only transitions with known event evidence. Roll the hold tail
    # into the final decision only when its physical terminal result is known.
    ids=np.flatnonzero(trusted[:-1]&trusted[1:])
    ids=ids[ids<active[-1]]
    if outcome_known and trusted[active[-1]]:
        ids=np.r_[ids,active[-1]]
    if require_replay and not len(ids):
        raise ValueError(f"{tr.path}: no eligible measured SAC transition")
    return dict(ids=ids,reward=reward,last=active[-1],outcome=outcome if outcome_known else None,
                trusted_frames=int(trusted.sum()))


def replay_item(tr, data, i):
    a=tr.a
    terminal=i==data["last"]
    j=len(a["t"])-1 if terminal else i+1
    def inputs(k):
        h,n=window(a["actor_frames"],a["t"],k,25)
        return dict(history=h,length=n,points=a["observed_points"][k],point_valid=a["observed_point_valid"][k],
                    point_conf=a["observed_point_conf"][k],physical=a["observed_physical"][k])
    r=data["reward"][i:j]
    return dict(current=inputs(i),next=inputs(j),z=a["selected_residual"][i],reward=float(r@(.99**np.arange(len(r)))),
                done=terminal,discount=.99**len(r))


def adaptation_agent(actor,dynamics,parent,args):
    args.lr,args.actor_lr,args.dynamics_lr=1e-4,1e-5,1e-4
    agent=SAC(actor,dynamics,args)
    agent.restore(parent,False)
    agent.log_alpha.data.copy_(parent["alpha"])
    actor.requires_grad_(False)
    actor.mu.requires_grad_(True)
    actor.log_std.requires_grad_(True)
    for q in agent.q:
        q.requires_grad_(False)
        q.head.requires_grad_(True)
    agent.q_params=[p for q in agent.q for p in q.head.parameters()]
    return agent


def adapt(a):
    cfg=config()
    identity=dict(parent_sha256=sha256(a.checkpoint),steps=a.steps,batch=a.batch,seed=a.seed,device=a.device,
                  code_sha256=sha256(Path(__file__)))
    actor,dyn,pre=load_models(a.checkpoint,a.device)
    calibration=pre.get("calibration_identity",{})
    if (not pre.get("identity",{}).get("deployment") or not pre.get("calibration_step")
            or pre["calibration_step"]!=calibration.get("steps")
            or "r1_step" in pre and pre["r1_step"]!=pre["r1_identity"]["steps"]):
        raise ValueError("R1 starts with deployed actor/critics and independently calibrated R0 dynamics")
    trials=[Trial(p,pre["joint_names"]) for p in a.receipt]
    expected_policy={identity["parent_sha256"]}
    if "r1_step" not in pre:
        # R0 only updates D0. Both the preceding S1 trials and new attempts
        # using the calibrated D0 checkpoint have the same behavior actor.
        expected_policy.add(calibration["parent"])
    if (len({r.meta["trial_id"] for r in trials})!=len(trials)
            or any(r.meta["split"]!="train" or r.meta["policy_sha256"] not in expected_policy for r in trials)):
        raise ValueError("R1 requires distinct train trials from the fixed deployed behavior checkpoint")
    identity["behavior_policy_sha256"]=sorted(expected_policy)
    identity["trials"]=[dict(receipt=str(r.path.resolve()),receipt_sha256=sha256(r.path),source_group_id=r.meta["source_group_id"],
                             trajectory_sha256=sha256(r.path.parent/r.meta["trajectory"])) for r in trials]
    replay=[(r,measured_returns(r,cfg)) for r in trials]
    policy_trials=[dict(trial=r.meta["trial_id"],known_frames=x["trusted_frames"],outcome=x["outcome"],
                        training_transitions=len(x["ids"])) for r,x in replay]
    torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    agent=adaptation_agent(actor,dyn,pre,a)
    checkpoint=a.out/"latest.pt"
    start=0
    if checkpoint.exists():
        z=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if not a.resume or z["r1_identity"]!=identity:
            raise ValueError("R1 resume requires unchanged receipts and budget")
        agent.restore(z)
        start=z["r1_step"]
        restore_rng(rng,z["r1_rng"])
    write(a.out/"config.json",identity)
    stop=[]
    for s in (signal.SIGTERM,signal.SIGINT):
        signal.signal(s,lambda *_:stop.append(True))
    for j in range(start,a.steps):
        data=[replay[k] for k in rng.integers(len(replay),size=a.batch)]
        b=gather([replay_item(t,x,int(rng.choice(x["ids"]))) for t,x in data],a.device)
        metrics=agent.update(b,[])
        if (j+1)%a.save_every==0 or j+1==a.steps or stop:
            save_checkpoint(checkpoint,dict(pre,**agent.state(),r1_identity=identity,r1_step=j+1,r1_rng=rng_state(rng)))
            print(dict(step=j+1,**metrics),flush=True)
        if stop:
            return
    write(a.out/"evaluation.json",dict(old_policy_measured_trials=policy_trials,updates=a.steps,
        checkpoint_sha256=sha256(checkpoint),dynamics_updated=False,
        accepted_training_receipts=[str(r.path.resolve()) for r,x in replay if x["outcome"] and x["outcome"]["success"]],
        new_policy_closed_loop="pending separate paired hardware trials"))


def compare_trials(a):
    """Assess a new actor with independent, same-condition hardware attempts."""
    cfg=config()
    chk=torch.load(a.checkpoint,map_location="cpu",weights_only=False,mmap=True)
    identity=chk.get("r1_identity")
    if not identity or chk.get("r1_step")!=identity["steps"]:
        raise ValueError("Compare after completing the declared R1 update budget")
    before_hashes=set(identity["behavior_policy_sha256"])
    train_groups={r["source_group_id"] for r in identity["trials"]}
    trial_ids=set()
    arms=[]
    for paths,weights in ((a.before_receipt,before_hashes),(a.receipt,{sha256(a.checkpoint)})):
        rows={}
        for path in paths:
            tr=Trial(path,chk["joint_names"])
            m=tr.meta
            group=m["source_group_id"]
            if (m["split"]!="test" or m["trial_id"] in trial_ids or m["policy_sha256"] not in weights
                    or group in train_groups or group in rows):
                raise ValueError(f"{path}: require one distinct held-out trial per condition and expected behavior checkpoint")
            if not m["complete"] or m.get("engineering_error"):
                raise ValueError(f"{path}: incomplete or engineering-invalid hardware attempt")
            metrics=measured_returns(tr,cfg,require_replay=False)
            rows[group]=dict(conditions={k:m[k] for k in ("sensor_calibration","actuator_calibration","task","phase_rules")},
                receipt=str(path.resolve()),receipt_sha256=sha256(path),policy_sha256=m["policy_sha256"],
                control_intervals=len(tr.dt),physical_seconds=float(tr.dt.sum()),outcome=metrics["outcome"],
                known_frames=metrics["trusted_frames"])
            trial_ids.add(m["trial_id"])
        arms.append(rows)
    before,after=arms
    if before.keys()!=after.keys() or any(before[g]["conditions"]!=after[g]["conditions"] for g in before):
        raise ValueError("Hardware comparison changed held-out conditions, calibrations or task")
    pairs=[dict(source_group_id=g,before=before[g],after=after[g]) for g in sorted(before)]
    known=all(r[k]["outcome"] is not None for r in pairs for k in ("before","after"))
    successes={k:sum(r[k]["outcome"]["success"] for r in pairs) for k in ("before","after")} if known else None
    write(a.out/"paired_hardware.json",dict(pairs=pairs,complete_witnesses=known,successes=successes,
        success_count_delta=successes["after"]-successes["before"] if known else None))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--receipt",type=Path,nargs="+",required=True)
    p.add_argument("--before-receipt",type=Path,nargs="+",help="Compare these old test attempts with new test receipts")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--save-every",type=int,default=25)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.before_receipt:
        if a.steps is not None or a.resume:
            p.error("Paired evaluation does not update or resume training")
    elif a.steps is None or min(a.steps,a.batch,a.save_every)<1:
        p.error("Use a finite positive R1 budget")
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/".run.lock").open("w") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (compare_trials if a.before_receipt else adapt)(a)

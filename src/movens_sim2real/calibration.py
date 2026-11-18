"""R0 offline D0 calibration on acknowledged hardware commands and masked targets."""
import argparse
import fcntl
import json
from pathlib import Path
import signal

import numpy as np
import torch

from .contract import EVENTS, read, sha256, split_map, state_delta, write
from .data import array_dict
from .features import DYN_SIZE, PRED_SIZE, window
from .learn import Replay
from .models import gather, load_models, save_checkpoint
from .pretrain import _event_metrics, ensemble, restore_rng, rng_state


FORMAT="fixed-base-robot-trial-v1"


class Trial:
    def __init__(self, path, names):
        self.path=path
        r=self.meta=read(path)
        if (r.get("format")!=FORMAT or r.get("source")!="recorded_hardware"
                or r.get("command_semantics")!="acknowledged_position_target_rad"):
            raise ValueError(f"{path}: calibrated hardware state and acknowledged applied commands are required")
        if not all(r.get(k) for k in ("capture_clock","sensor_calibration","actuator_calibration",
                                      "applied_command_source","trial_id","source_group_id")):
            raise ValueError(f"{path}: missing trial identity, common clock or command/calibration provenance")
        a=self.a=array_dict(Path(path).parent/r["trajectory"])
        t=a["t"]
        n=len(t)
        dt=np.diff(t)
        applied=a["applied_start_s"]
        end=a["applied_end_s"]
        if (n<2 or a["joint_names"].tolist()!=list(names) or t.ndim!=1 or not np.isfinite(t).all()
                or np.any(abs(dt-.02)>.002) or applied.shape!=(n-1,) or not np.isfinite(applied).all()
                or np.any((applied<t[:-1])|(applied-t[:-1]>.003)|(applied>=t[1:]))
                or end.shape!=(n-1,) or not np.isfinite(end).all()
                or np.any(end<t[1:]-1e-8) or np.any(end-t[1:]>.003)
                or np.any(abs(end[:-1]-applied[1:])>2e-4)
                or a["executed"].shape!=(n-1,43) or not np.isfinite(a["executed"]).all()
                or len(np.unique(a["reset_id"]))!=1 or len(a["reset_id"])!=n
                or a["dynamics_frames"].shape!=(n,3,DYN_SIZE)):
            raise ValueError(f"{path}: transition clock, reset boundary or joint order is incompatible")
        for key,shape in (("dynamics_frames",(n,3,DYN_SIZE)),("observed_physical",(n,3,29)),
                          ("observed_points",(n,256,4)),("observed_point_valid",(n,256)),
                          ("observed_point_conf",(n,256))):
            if a[key].shape!=shape or not np.isfinite(a[key]).all():
                raise ValueError(f"{path}: invalid deployed {key}")
        if (a["observed_point_valid"].dtype!=bool or np.any((a["observed_point_conf"]<0)|(a["observed_point_conf"]>1))
                or np.any(a["observed_point_conf"][~a["observed_point_valid"]])):
            raise ValueError(f"{path}: inconsistent partial geometry validity/confidence")
        m=a["measured_valid"]
        cf=a["measured_confidence"]
        if (m.dtype!=bool or m.shape!=(n,PRED_SIZE) or cf.shape!=m.shape
                or not np.isfinite(cf).all() or np.any((cf<0)|(cf>1))):
            raise ValueError(f"{path}: invalid per-dimension hardware evidence")
        m=m&(cf>=.6)
        states=[np.asarray(a[k]) for k in ("q","qd","object","twist")]
        if [x.shape for x in states]!=[(n,43),(n,43),(n,7),(n,6)]:
            raise ValueError(f"{path}: expected fixed-base q/qd, xyz/wxyz, and SI twist")
        rot=m[:,89:92].all(1)
        if np.any(abs(np.linalg.norm(states[2][rot,3:],axis=1)-1)>1e-5):
            raise ValueError(f"{path}: trusted orientation must be calibrated unit wxyz")
        ms=m.copy()
        ms[:,89:92]=rot[:,None]
        state_mask=[ms[:,:43],ms[:,43:86],np.c_[ms[:,86:89],np.repeat(rot[:,None],4,axis=1)],ms[:,92:]]
        if any(not np.isfinite(x[y]).all() for x,y in zip(states,state_mask)):
            raise ValueError(f"{path}: nonfinite trusted hardware state")
        states=[np.where(y,x,0.) for x,y in zip(states,state_mask)]
        states[2][~rot,3:]=[1.,0.,0.,0.]
        self.delta=state_delta(*(x[:-1] for x in states),*(x[1:] for x in states))
        self.valid=ms[:-1]&ms[1:]
        self.ids=np.flatnonzero(self.valid.any(1))
        ev=a["events"]
        if (not len(self.ids) or ev.shape!=(n,5) or not np.isin(ev,[0,1]).all()
                or a["events_valid"].shape!=ev.shape or a["events_valid"].dtype!=bool):
            raise ValueError(f"{path}: no trusted transition or missing binary event provenance")
        if a["events_valid"].any() and not r.get("event_evidence_source"):
            raise ValueError(f"{path}: observable event labels require an independent evidence source")
        self.dt=dt
        self.relative_delay=applied-t[:-1]

    def item(self,i,history):
        a=self.a
        h,n=window(a["dynamics_frames"],a["t"],i,history)
        return dict(history=h,length=n,points=a["observed_points"][i],
            point_valid=a["observed_point_valid"][i],point_conf=a["observed_point_conf"][i],
            physical=a["observed_physical"][i],action=a["executed"][i],dt=[self.dt[i]],
            delta=self.delta[i],delta_valid=self.valid[i],events=a["events"][i+1].astype(np.float32),
            events_valid=a["events_valid"][i+1])


def report(models, trials, args):
    errors,mask,event,y,valid=[],[],[],[],[]
    with torch.inference_mode():
        for tr in trials:
            for first in range(0,len(tr.ids),args.batch):
                items=[tr.item(i,25) for i in tr.ids[first:first+args.batch]]
                batch=gather(items,args.device)
                mu,_,_,p=ensemble(models,batch)
                errors.append(np.abs(batch["delta"].cpu().numpy()-mu.cpu().numpy()))
                mask.append(batch["delta_valid"].cpu().numpy())
                event.append(p.cpu().numpy())
                y.append(batch["events"].cpu().numpy())
                valid.append(batch["events_valid"].cpu().numpy())
    e,m,p,y,v=map(np.concatenate,(errors,mask,event,y,valid))
    summary={}
    for tag,sel in (("q_rad",slice(0,43)),("qd_rad_s",slice(43,86)),("object_m",slice(86,89)),
                    ("rotation_rad",slice(89,92)),("linear_m_s",slice(92,95)),("angular_rad_s",slice(95,98))):
        x=e[:,sel][m[:,sel]]
        summary[tag]=dict(n=int(len(x)),mean_abs=float(x.mean()) if len(x) else None,
                          p95_abs=float(np.quantile(x,.95)) if len(x) else None)
    return dict(trials=len(trials),control_transitions=len(e),delta=summary,
                events={k:_event_metrics(p[:,i],y[:,i],v[:,i]) for i,k in enumerate(EVENTS)})


def calibrate(a):
    name_state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    trials=[Trial(p,name_state["joint_names"]) for p in a.receipt]
    parent=sha256(a.checkpoint)
    if any(r.meta.get("policy_sha256")!=parent for r in trials):
        raise ValueError("R0 trials must identify the fixed deployment checkpoint that supplied their actions")
    if len({r.meta["trial_id"] for r in trials})!=len(trials):
        raise ValueError("R0 trial IDs must be unique across declared receipts")
    train=[r for r in trials if r.meta["split"]=="train"]
    held=[r for r in trials if r.meta["split"]=="test"]
    groups={s:{r.meta["source_group_id"] for r in trials if r.meta["split"]==s} for s in ("train","test")}
    split_map(groups)
    if not train or not held or len(train)+len(held)!=len(trials):
        raise ValueError("R0 requires independent assigned hardware train and test trials")
    if not name_state.get("identity",{}).get("deployment"):
        raise ValueError("R0 starts from the deployed actor/critic and D0 checkpoint")
    del name_state
    _,models,x=load_models(a.checkpoint,a.device)
    optim=[torch.optim.AdamW(m.parameters(),lr=a.lr) for m in models]
    torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    sim=Replay(25,.99)
    sim_inputs=sorted(p for root in a.sim for p in root.glob("train/*/summary.json"))
    for p in sim_inputs:
        sim.add(read(p))
    if a.sim and not sim.episodes:
        raise ValueError("Requested sim mix has no measured source transitions")
    identity=dict(parent=parent,recordings=[(str(r.path.resolve()),sha256(r.path),
        sha256(r.path.parent/r.meta["trajectory"])) for r in trials],sim=[(str(p.resolve()),sha256(p)) for p in sim_inputs],
        steps=a.steps,batch=a.batch,seed=a.seed,lr=a.lr,device=a.device,code=sha256(Path(__file__)),
        mixture="half real + half sim" if a.sim else "real only")
    out=a.out
    ck=out/"latest.pt"
    before_path=out/"before.json"
    step=0
    if ck.exists():
        z=torch.load(ck,map_location="cpu",weights_only=False)
        if not a.resume or json.loads(json.dumps(identity))!=z["calibration_identity"]:
            raise ValueError("R0 checkpoint inputs or budget changed")
        for m,o,w,s in zip(models,optim,z["dynamics"],z["calibration_optimizers"]):
            m.load_state_dict(w); o.load_state_dict(s)
        step=z["calibration_step"]
        restore_rng(rng,z["calibration_rng"])
    else:
        write(out/"config.json",identity)
        write(before_path,report(models,held,a))
    stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.append(True))
    for j in range(step,a.steps):
        losses=[]
        for model,opt in zip(models,optim):
            model.train()
            nr=a.batch//2 if a.sim else a.batch
            samples=[t.item(int(rng.choice(t.ids)),25) for t in (train[i] for i in rng.integers(len(train),size=nr))]
            real=gather(samples,a.device)
            if a.sim:
                replay=sim.sample(rng,a.batch-nr,a.device,dynamics=True,clean_fraction=.5)
                real={k:torch.cat((v,replay[k]),0) for k,v in real.items()}
            loss=model.loss(real)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite real calibration loss at {j+1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
        if (j+1)%a.save_every==0 or j+1==a.steps or stop:
            save_checkpoint(ck,dict(x,dynamics=[m.state_dict() for m in models],
                calibration_identity=json.loads(json.dumps(identity)),calibration_optimizers=[o.state_dict() for o in optim],
                calibration_step=j+1,calibration_rng=rng_state(rng)))
            print(f"R0 step={j+1} losses={losses}",flush=True)
        if stop:
            return
    for m in models:
        m.eval()
    timing={s:dict(state_dt_p05_p50_p95=np.quantile(np.concatenate([t.dt for t in rs]),[.05,.5,.95]),
                  state_to_applied_p05_p50_p95=np.quantile(np.concatenate([t.relative_delay for t in rs]),[.05,.5,.95]))
            for s,rs in (("train",train),("test",held))}
    write(out/"evaluation.json",dict(before=read(before_path),after=report(models,held,a),time_seconds=timing,
                                    actor_changed=False,dynamics_controls_action=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--receipt",type=Path,nargs="+",required=True)
    p.add_argument("--sim",type=Path,nargs="*",default=[],help="Optional real MuJoCo training replay roots")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,required=True)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--save-every",type=int,default=25)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.batch<2 or min(a.steps,a.save_every,a.lr)<=0:
        p.error("Choose a finite calibration budget")
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/".run.lock").open("w") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        calibrate(a)


if __name__=="__main__":
    main()

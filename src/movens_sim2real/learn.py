"""Finite-budget D0 SAC training on actual immutable episode logs."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from .contract import ROOT, SCHEMA, config, read, sha256, state_delta, write
from .data import array_dict, split_apple
from .features import window
from .models import ARCH, Actor, Q, gather, load_models, save_checkpoint
from .pretrain import ensemble, rng_state, restore_rng
from .rollout import episode_rollout, file_identity, policy_load


class EpisodeTransitions:
    def __init__(self, receipt, history, gamma):
        self.receipt=receipt
        self.path=Path(receipt["path"])
        self.split=receipt["identity"]["split"]
        self.domain="observed" if receipt["identity"]["adapter"] is not None else "clean"
        a=self.a=array_dict(self.path)
        self.history=history
        self.indices=np.flatnonzero(a["decision_mask"])
        self.dyn_ids=np.flatnonzero(a["actual_20ms"])
        self.nearly_event=np.intersect1d(self.dyn_ids,np.unique((np.flatnonzero(
            np.any(a["events"][1:]!=a["events"][:-1],axis=1))[:,None]+np.arange(-3,4)).ravel()))
        n=len(a["t"])
        if (str(a["schema"])!=SCHEMA or n-1!=len(a["executed"]) or self.split!="train"
                or len(self.indices)==0 or not np.array_equal(a["actuator"][:,a["actuator_indices"]],a["executed"])
                or np.any(np.diff(a["t"])<=0) or not np.allclose(np.diff(a["t"]),.02,rtol=0,atol=1e-8)
                or not np.array_equal(a["tick"],np.arange(n-1))):
            raise ValueError(f"{self.path}: invalid replay transitions")
        if not np.array_equal(self.indices,np.arange(len(self.indices))):
            raise ValueError("Decision mask includes a reset or non-contiguous policy interval")
        self.reward=a["reward"].copy()
        self.done=a["terminated"].copy()
        self.next_index=self.indices+1
        self.discount=np.full(len(self.indices),gamma,dtype=np.float32)
        terminal_tail=self.indices[-1]
        if terminal_tail+1<n-1 and a["terminated"][-1]:
            # The hold tail has no actions sampled from the policy. Combine
            # its physical rewards with the last true decision (semi-MDP).
            r=a["reward"][terminal_tail:]
            self.reward[terminal_tail]=np.dot(r,gamma**np.arange(len(r)))
            self.done[terminal_tail]=a["terminated"][-1]
            self.next_index[-1]=len(a["t"])-1
            self.discount[-1]=gamma**len(r)
        elif terminal_tail+1<n-1:
            # No policy decision follows the last active action when a budget
            # cuts the fixed verification tail. The incomplete return is
            # neither terminal nor a valid policy bootstrap point. The full
            # logged intervals still supervise the D0 dynamics.
            self.indices=self.indices[:-1]
        k=("q","qd","object","twist")
        self.delta=state_delta(*(a[x][:-1] for x in k),*(a[x][1:] for x in k)).astype(np.float32)

    def context(self,i,key):
        a=self.a
        h,n=window(a[key+"_frames"],a["t"],i,self.history)
        return dict(history=h,length=n,points=a["observed_points"][i],
                    point_valid=a["observed_point_valid"][i],point_conf=a["observed_point_conf"][i],
                    physical=a["observed_physical"][i])

    def rl(self,i):
        j=self.next_index[i]
        return dict(current=self.context(i,"actor"),next=self.context(j,"actor"),
                    z=self.a["selected_residual"][i],reward=self.reward[i],done=self.done[i],discount=self.discount[i])

    def dynamic(self,i):
        a=self.a
        return dict(**self.context(i,"dynamics"),action=a["executed"][i],dt=[a["t"][i+1]-a["t"][i]],
                    delta=self.delta[i],delta_valid=np.ones(98,bool),events=a["events"][i+1].astype(np.float32),
                    events_valid=a["events_valid"][i+1])


class Replay:
    def __init__(self,history,gamma):
        self.history=history
        self.gamma=gamma
        self.episodes=[]

    def add(self,rec):
        if rec["engineering_error"] is not None:
            return
        self.episodes.append(EpisodeTransitions(rec,self.history,self.gamma))

    @property
    def n(self):
        return sum(len(ep.indices) for ep in self.episodes)

    def sample(self,rng,batch,device,*,dynamics=False,clean_fraction=None):
        tables=[self.episodes]
        n=[batch]
        if clean_fraction is not None:
            clean=[x for x in self.episodes if x.domain=="clean"]
            changed=[x for x in self.episodes if x.domain!="clean"]
            if clean and changed:
                nc=round(clean_fraction*batch)
                tables,n=[clean,changed],[nc,batch-nc]
        out=[]
        for es,nn in zip(tables,n):
            probs=np.asarray([len(ep.dyn_ids if dynamics else ep.indices) for ep in es],dtype=np.float64)
            for j in rng.choice(len(es),size=nn,p=probs/probs.sum()):
                ep=es[j]
                opts=(ep.nearly_event if len(ep.nearly_event) and rng.random()<.5 else ep.dyn_ids) if dynamics else ep.indices
                i=int(rng.choice(opts))
                out.append(ep.dynamic(i) if dynamics else ep.rl(i))
        return gather(out,device)


def _optimize(optim,loss,parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite SAC/dynamics objective")
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
    optim.step()


class SAC:
    def __init__(self,actor,dynamics,args):
        self.actor,self.dynamics=actor.train(),[m.train() for m in dynamics]
        self.q=[Q().to(args.device) for _ in range(2)]
        for m in self.q:
            m.encoder.norm.load_state_dict(actor.encoder.norm.state_dict())
            m.phys.load_state_dict(dynamics[0].phys.state_dict())
        self.targets=[deepcopy(m).requires_grad_(False).eval() for m in self.q]
        self.log_alpha=torch.tensor(np.log(.05),device=args.device,dtype=torch.float32,requires_grad=True)
        self.actor_optim=torch.optim.Adam(actor.parameters(),lr=args.actor_lr)
        self.q_params=[p for m in self.q for p in m.parameters()]
        self.q_optim=torch.optim.Adam(self.q_params,lr=args.lr)
        self.temp_optim=torch.optim.Adam([self.log_alpha],lr=args.lr)
        self.d_optim=[torch.optim.Adam(m.parameters(),lr=args.dynamics_lr) for m in self.dynamics]
        self.updates=0

    def update(self, b, db, actor_frozen=False, critic_only=False):
        actor=self.actor
        alpha=self.log_alpha.detach().exp()
        with torch.no_grad():
            z,lp=actor.sample(b["next"])
            v=torch.minimum(*(q(b["next"],z) for q in self.targets))-alpha*lp
            y=b["reward"]+b["discount"]*(~b["done"])*v
        q_loss=sum(F.mse_loss(q(b["current"],b["z"]),y) for q in self.q)
        _optimize(self.q_optim,q_loss,self.q_params)
        a_loss=None
        if not critic_only and not actor_frozen:
            for p in self.q_params:
                p.requires_grad_(False)
            z,lp=actor.sample(b["current"])
            a_loss=(alpha*lp-torch.minimum(*(m(b["current"],z) for m in self.q))).mean()
            _optimize(self.actor_optim,a_loss,actor.parameters())
            for p in self.q_params:
                p.requires_grad_(True)
            entropy=-(self.log_alpha*(lp.detach()-7.)).mean()
            _optimize(self.temp_optim,entropy,[self.log_alpha])
        with torch.no_grad():
            for q,qt in zip(self.q,self.targets):
                for p,d in zip(q.parameters(),qt.parameters()):
                    d.lerp_(p,.005)
            self.log_alpha.clamp_(-10.,2.)
        losses=[]
        for m,opt,d in zip(self.dynamics,self.d_optim,db):
            l=m.loss(d)
            _optimize(opt,l,m.parameters())
            losses.append(float(l.detach()))
        self.updates+=1
        return dict(q=float(q_loss.detach()), actor=float(a_loss.detach()) if a_loss is not None else None,
                    alpha=float(self.log_alpha.detach().exp()),dynamics=losses,
                    target_mean=float(y.mean()),updates=self.updates)

    def state(self):
        return dict(arch=ARCH,actor=self.actor.state_dict(),dynamics=[m.state_dict() for m in self.dynamics],
                    critics=[q.state_dict() for q in self.q],targets=[q.state_dict() for q in self.targets],
                    alpha=self.log_alpha.detach(),actor_optim=self.actor_optim.state_dict(),q_optim=self.q_optim.state_dict(),
                    temp_optim=self.temp_optim.state_dict(),dynamics_optim=[o.state_dict() for o in self.d_optim],updates=self.updates)

    def restore(self,z, resume=True):
        self.actor.load_state_dict(z["actor"])
        for models,key in ((self.dynamics,"dynamics"),(self.q,"critics"),(self.targets,"targets")):
            for m,state in zip(models,z[key]):
                m.load_state_dict(state)
        if resume:
            self.log_alpha.data.copy_(z["alpha"])
            for opt,key in ((self.actor_optim,"actor_optim"),(self.q_optim,"q_optim"),(self.temp_optim,"temp_optim")):
                opt.load_state_dict(z[key])
            for opt,state in zip(self.d_optim,z["dynamics_optim"]):
                opt.load_state_dict(state)
            self.updates=z["updates"]


def stat_rows(rows):
    outcomes=[r for r in rows if r is not None]
    n=len(outcomes)
    return dict(episodes=n, complete=sum(r["complete"] for r in outcomes),
                engineering_errors=sum(r["engineering_error"] is not None for r in outcomes),
                success=sum(r["success"] for r in outcomes),
                verified_grasps=sum(r["stable_grasp"] for r in outcomes),
                transported=sum(r["transported"] for r in outcomes), slips=sum(r["slip_count"]>0 for r in outcomes),
                stable_releases=sum(r["stable_release_s"]>=1.-1e-8 for r in outcomes),
                export_eligible=sum(r["success_export_eligible"] for r in outcomes),
                control_ticks=sum(r["decision_steps"] for r in outcomes),
                policy_calls=sum(r["policy_calls"] for r in outcomes),
                reasons=Counter(r["failure"] for r in outcomes if r["failure"]))


@torch.no_grad()
def online_dynamics_summary(rows):
    errors,p,ys,valid=[],[],[],[]
    for row in rows:
        if not row or row["engineering_error"]:
            continue
        a=array_dict(row["path"])
        if "predicted_delta" not in a:
            continue
        k=("q","qd","object","twist")
        d=state_delta(*(a[x][:-1] for x in k),*(a[x][1:] for x in k))
        errors.append(d-a["predicted_delta"])
        p.append(a["predicted_events"])
        ys.append(a["events"][1:])
        valid.append(a["events_valid"][1:])
    if not errors:
        return {}
    from .pretrain import summarize_delta, _event_metrics
    from .contract import EVENTS
    e,p,y,v=map(np.concatenate,(errors,p,ys,valid))
    return dict(**summarize_delta(e),event={k:_event_metrics(p[:,i],y[:,i],v[:,i]) for i,k in enumerate(EVENTS)})


def train(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    cfg=config(args.config)
    groups=split_apple(cfg)
    if any(groups[i]!="train" for i in args.train) or any(groups[i]=="train" for i in args.eval):
        raise ValueError("SAC training and evaluation source groups must be separated")
    if len(set(args.eval))!=len(args.eval) or not args.train:
        raise ValueError("Duplicate evaluation source or empty training sequence")
    history=args.history or cfg["history"]
    identity={k:v for k,v in vars(args).items() if k not in ("resume","out","stop_after_updates")}
    code=("learn.py","models.py","rollout.py","features.py")
    if args.deployment:
        code+=("deployment.py","hardware.py","vision.py","scene.py","phase.py")
    clean_paths=sorted(args.clean_replay.glob("train/*/summary.json")) if args.clean_replay else []
    identity.update(schema=SCHEMA,arch=ARCH,execution=dict(config=sha256(args.config),pretrained=sha256(args.pretrained),
                    code={name:sha256(Path(__file__).with_name(name)) for name in code}))
    if args.deployment:
        identity["execution"].update(base=file_identity(args.base_adapter or cfg["checkpoint"]),
            clean_replay=[dict(path=str(p.resolve()),summary_sha256=sha256(p),
                               trajectory_sha256=sha256(read(p)["path"])) for p in clean_paths])
    receipt=out/"config.json"
    from .contract import _default
    if receipt.exists():
        if not args.resume or read(receipt)!=json.loads(json.dumps(identity,default=_default)):
            raise ValueError("SAC resume must preserve exact budget and input identity")
    else:
        write(receipt,identity)
    identity=read(receipt)
    stop=[]
    for s in (signal.SIGTERM,signal.SIGINT):
        signal.signal(s,lambda *_:stop.append(True))
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    actor,dyn,pre=load_models(args.pretrained,args.device)
    if args.cold:
        actor=Actor().to(args.device)
        actor.encoder.norm.mean.copy_(pre["actor"]["encoder.norm.mean"])
        actor.encoder.norm.std.copy_(pre["actor"]["encoder.norm.std"])
    agent=SAC(actor,dyn,args)
    if args.deployment:
        if "critics" not in pre:
            raise ValueError("S1 must branch from a clean S0 SAC checkpoint")
        agent.restore(pre,False)
        agent.log_alpha.data.copy_(pre["alpha"])
    policy=policy_load(cfg,args.device,args.base_adapter)
    replay=Replay(history,args.gamma)
    if args.deployment:
        for pp in clean_paths:
            clean=read(pp)
            if not clean["complete"] or clean["identity"]["adapter"] is not None or clean["engineering_error"]:
                raise ValueError(f"{pp}: S1 clean replay must come from a complete clean S0 rollout")
            replay.add(clean)
        if not replay.episodes:
            raise ValueError("S1 blend requires clean S0 replay")
    idx,steps,pending=0,0,0
    checkpoint=out/"latest.pt"
    previous_rows=[]
    if checkpoint.exists():
        x=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if x["identity"]!=identity:
            raise ValueError("SAC optimizer checkpoint identity differs")
        idx,steps,pending=x["episode_index"],x["env_control_steps"],x["pending_updates"]
        previous_rows=x["train_receipts"]
        agent.restore(x)
        for path in previous_rows:
            replay.add(read(path))
        restore_rng(rng,x["rng"])
    wall=time.monotonic()
    def snapshot():
        save_checkpoint(checkpoint,dict(agent.state(),identity=identity,episode_index=idx,env_control_steps=steps,
                pending_updates=pending,train_receipts=previous_rows,rng=rng_state(rng),joint_names=pre["joint_names"]))
    if not checkpoint.exists():
        snapshot()
    cfrac=.5 if args.deployment else None
    while True:
        last_results=None
        while pending and not stop:
            b=replay.sample(rng,args.batch,args.device,clean_fraction=cfrac)
            db=[replay.sample(rng,args.batch,args.device,dynamics=True,clean_fraction=cfrac) for _ in dyn]
            last_results=agent.update(b,db)
            pending-=1
            reached_pause=args.stop_after_updates is not None and agent.updates>=args.stop_after_updates
            if agent.updates%args.save_every==0 or pending==0 or reached_pause:
                snapshot()
                with (out/"metrics.jsonl").open("a") as log:
                    print(json.dumps(dict(last_results,replay=replay.n,env_steps=steps,duration=time.monotonic()-wall)),
                          file=log)
                print(f"SAC updates={agent.updates} env_steps={steps} replay={replay.n} {last_results}",flush=True)
            if reached_pause:
                return
        if stop:
            snapshot()
            return
        if steps>=args.steps or idx>=len(args.train):
            break
        ep=args.train[idx]
        if args.steps-steps<5:
            break
        root=out/"train"/f"{idx:03d}_apple_{ep:06d}"
        histories=dict(updates=agent.updates,origin_sha256=identity["execution"]["pretrained"],experiment=out.name)
        for m in (actor,*dyn):
            m.eval()
        # A clean S1 sequence from the held simulator is mixed explicitly at
        # replay sampling; all newly collected S1 inputs go through the adapter.
        adapter=None
        if args.deployment:
            from .deployment import factory
            adapter=factory(args.deployment,idx)
        rec=episode_rollout(cfg,ep,root,policy,actor,dyn,small_identity=histories,seed=args.seed,train=True,
                            trial=idx,history_length=history,decision_limit=args.steps-steps,adapter_factory=adapter,
                            phase_mode=args.phase,stop=lambda:bool(stop))
        if rec is None:
            snapshot()
            return
        steps+=rec["decision_steps"]
        idx+=1
        if not rec["engineering_error"]:
            replay.add(rec)
            previous_rows.append(root/"summary.json")
            if replay.n>=args.warmup:
                pending=min(max(0,args.max_updates-agent.updates),round(rec["active_steps"]*args.update_ratio))
        snapshot()
        for m in (actor,*dyn):
            m.train()
    if agent.updates==0:
        raise ValueError("Budget consumed without a real SAC update")
    save_checkpoint(out/"final.pt",dict(agent.state(),identity=identity,joint_names=pre["joint_names"],
                                       env_control_steps=steps,train_receipts=previous_rows))
    for m in (actor,*dyn):
        m.eval()
    final=file_identity(out/"final.pt")
    # Use update number rather than last.pt mtime as rollout identity.
    final={k:final[k] for k in ("path","size")}
    result=dict(train=stat_rows([read(p) for p in previous_rows]),env_steps=steps,updates=agent.updates,
                test_splits={})
    write(out/"evaluation.json",result)
    modes=("clean","physics","noisy","deployment") if args.deployment else ("clean",)
    for mode in modes if args.eval else ():
        adapter=None
        if mode!="clean":
            from .deployment import factory
            adapter=factory(mode)
        rows=[]
        for i in args.eval:
            if stop:
                return
            cur=episode_rollout(cfg,i,out/"eval"/mode/f"apple_{i:06d}",policy,actor,dyn,small_identity=final,
                seed=args.seed,side=args.eval_side,history_length=history,phase_mode=args.phase,adapter_factory=adapter,
                stop=lambda:bool(stop))
            if cur is None:
                return
            rows.append(cur)
        result["test_splits"][mode]=dict(stat_rows(rows),d0=online_dynamics_summary(rows),
                                         episodes=[r["identity"]["episode"] for r in rows])
        write(out/"evaluation.json",result)
    print(f"Finished SAC: {steps} native control transitions, {agent.updates} optimizer updates",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/experiment.json")
    p.add_argument("--pretrained",type=Path,required=True)
    p.add_argument("--base-adapter",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--train",type=int,nargs="+",required=True)
    p.add_argument("--eval",type=int,nargs="*",required=True)
    p.add_argument("--eval-side",action="store_true",help="Use the same dual-view rendering protocol as the frozen baseline")
    p.add_argument("--steps",type=int,required=True)
    p.add_argument("--max-updates",type=int,required=True)
    p.add_argument("--warmup",type=int,default=500)
    p.add_argument("--update-ratio",type=float,default=1.)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--history",type=int,choices=[1,8,25])
    p.add_argument("--phase",choices=["phase","time","none"],default="phase")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--actor-lr",type=float,default=1e-4)
    p.add_argument("--dynamics-lr",type=float,default=1e-4)
    p.add_argument("--gamma",type=float,default=.99)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--cold",action="store_true")
    p.add_argument("--deployment",choices=["deployment","curriculum"])
    p.add_argument("--clean-replay",type=Path)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--stop-after-updates",type=int,help="Save and return at this absolute optimizer update")
    a=p.parse_args()
    if (a.steps<=0 or a.max_updates<=0 or a.update_ratio<=0 or a.history is not None and a.history<=0
            or bool(a.deployment)!=bool(a.clean_replay)
            or a.stop_after_updates is not None and not 0<a.stop_after_updates<=a.max_updates):
        p.error("Invalid budget or missing clean-replay input for deployment")
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/".run.lock").open("w") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train(a)


if __name__=="__main__":
    main()

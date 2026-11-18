"""Train both small models with genuine transitions and SONIC state history."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .contract import ActionLimits, LEFT, ROOT, SCHEMA, config, read, rotation, sha256, split_map, state_delta, write, wxyz
from .data import array_dict, obs_at
from .features import ACTOR_SIZE, DYN_SIZE, actor_frame, dyn_frame, packed, point_context, window
from .models import ARCH, Actor, Dynamics, gather, save_checkpoint, stats, tensors


@dataclass
class Episode:
    file: Path
    identity: dict
    a: dict
    length: int

    @property
    def split(self):
        return self.identity["split"]

    def actor(self, mode):
        a = self.a
        limits = ActionLimits(*(a[name] for name in ("action_lo","action_hi","residual_scale","residual_max_step")))
        targets = a["all_commands"].copy()
        original = targets[:, LEFT].copy()
        seed = int.from_bytes(hashlib.sha256((self.identity["source_group_id"]+mode).encode()).digest()[:8],"little")
        if mode == "fixed":
            lo, hi = limits.lo[LEFT], limits.hi[LEFT]
            # A displacement along the actuator range, slew-constrained as
            # in the real interface. This ablation has no stochastic error.
            desired = np.clip(.5*original, lo, hi)
            previous = np.zeros(7)
            adjusted = []
            for t, n in zip(original, desired):
                e = np.clip(t-n, previous-limits.max_step, previous+limits.max_step)
                e = np.clip(e,-limits.scale,limits.scale)
                n = np.clip(t-e,lo,hi)
                previous = t-n
                adjusted.append(n)
            nominal = np.asarray(adjusted)
            label = (original-nominal)/limits.scale
            used = "fixed"
        else:
            nominal,label,used = split_residual(original,limits,np.random.default_rng(seed),mode)
        targets[:, LEFT] = nominal
        close = []
        xv = []
        dt = .02
        t = a["t"]
        idx = np.flatnonzero(t < float(a["motion_end_s"])-1e-9)
        for i in idx:
            k = int(a["control_ticks"][i])
            end = min(len(targets), k//20*20+30)
            prev = a["all_commands"][k-1] if k else a["initial_target"]
            corr = original[k-1]-nominal[k-1] if k else None
            phase = (int(a["phase_id"][i]), a["phase_valid"][i], a["phase_confidence"][i], a["transition_reason"][i],
                     a["phase_age_s"][i], a["stable_release_s"][i], a["slip"][i])
            ob = obs_at(a,i)
            ob["time"] = t[i]
            xv.append(actor_frame(ob,targets[k:end],a["nominal_wrist"][k:end],prev,corr,phase,
                                  float(a["motion_end_s"])-t[i],(k%20)*dt))
            close.append(label[k])
        return ActorEpisode(self, np.asarray(xv), np.asarray(close, dtype=np.float32), used)

    def dynamic(self):
        a = self.a
        if "sampled_prev_target" in a:
            prev = a["sampled_prev_target"]
        else:
            prev = np.vstack((a["initial_target"],a["executed"]))
        hist = np.asarray([dyn_frame(obs_at(a, i),prev[i]) for i in range(len(a["t"]))])
        keys = ["q","qd","object","twist"]
        delta = state_delta(*(a[k][:-1] for k in keys),*(a[k][1:] for k in keys))
        valid = np.flatnonzero(a["actual_20ms"])
        if not len(valid):
            return None
        ix = a["control_ticks"][:-1][valid]
        if (not np.allclose(np.diff(a["t"])[valid],.02,rtol=0,atol=1e-8)
                or not np.array_equal(a["all_commands"][ix],a["executed"][valid])
                or not np.array_equal(a["actuator"][valid][:,a["actuator_indices"]],a["executed"][valid])):
            raise ValueError(f"{self.file}: physical transition and applied target do not match")
        change = np.flatnonzero(np.any(a["events"][1:]!=a["events"][:-1],axis=1))
        neighbors = np.unique((change[:,None]+np.arange(-3,4)).ravel())
        focused = np.intersect1d(valid,neighbors)
        return DynEpisode(self,hist,delta.astype(np.float32),valid,focused)

    def context(self):
        a = self.a
        return point_context(a["points"])

    def physical(self):
        return packed(self.a["physical"],self.a["physical_valid"])


from .contract import split_residual


@dataclass
class ActorEpisode:
    episode: Episode
    x: np.ndarray
    y: np.ndarray
    mode: str

    def item(self,i, variant="full"):
        ep = self.episode
        h,n = window(self.x,ep.a["t"],i,ep.length)
        geo = ep.context()
        if variant == "nominal":
            h[..., :-51] = 0.
            geo["point_valid"][:] = False
            geo["point_conf"][:] = 0.
            h = np.roll(h, -(n-1), axis=0)
            n = np.int64(1)
        return dict(history=h,length=n,label=self.y[i],**geo)


@dataclass
class DynEpisode:
    episode: Episode
    x: np.ndarray
    delta: np.ndarray
    valid: np.ndarray
    focused: np.ndarray

    def item(self,i):
        ep,a = self.episode,self.episode.a
        h,n = window(self.x,a["t"],i,ep.length)
        return dict(history=h,length=n,physical=ep.physical(), action=a["executed"][i], dt=[a["t"][i+1]-a["t"][i]],
                    delta=self.delta[i], delta_valid=np.ones(98,dtype=bool), events=a["events"][i+1].astype(np.float32),
                    events_valid=a["events_valid"][i+1], **ep.context())


def records(root, length, pilot=False):
    paths = list((root/"sparse").glob("*.json"))
    paths += list((root/"dense").glob("apple_*/summary.json"))
    table = []
    split_sets = {s: set() for s in ("train","val","test")}
    for p in sorted(paths):
        meta = read(p)
        if not pilot and meta["failure"] in ("incomplete_or_local_reset","engineering_invalid"):
            continue
        a = array_dict(meta["path"])
        if str(a["schema"]) != SCHEMA:
            raise ValueError(f"{p}: data schema differs")
        d = meta["identity"]
        split_sets[d["split"]].add(d["source_group_id"])
        table.append((Episode(Path(meta["path"]),d,a,length),meta))
    split_map(split_sets)
    if not table:
        raise ValueError(f"{root}: no indexed data")
    names = table[0][0].a["joint_names"]
    for ep,_ in table:
        if not np.array_equal(ep.a["joint_names"],names):
            raise ValueError("Joint-mapped episode order differs")
    return table


class Sonic:
    def __init__(self, manifest, length):
        self.tables = {s: [] for s in ("train","val","test")}
        self.length = length
        for r in read(manifest)["records"]:
            z = array_dict(r["path"])
            self.tables[r["split"]].append(z["q"])
        if not all(self.tables.values()):
            raise ValueError("Missing SONIC group split")

    def batch(self,rng,n,width,device,split="train"):
        episodes = self.tables[split]
        samples = []
        for i in rng.integers(len(episodes),size=n):
            q = episodes[i]
            t = rng.integers(len(q)-1)
            a = q[max(0,t+1-self.length):t+1]
            x = np.zeros((self.length,3,width),dtype=np.float32)
            x[:len(a),0,:43] = a
            x[:len(a),1:,:43] = 1.
            samples.append(dict(history=x,length=np.int64(len(a)), label=(q[t+1]-q[t])/.05,
                                 **point_context(np.zeros((256,4)),np.zeros(256,bool))))
        return gather(samples,device)


def choose(es, rng, n, dynamics=False):
    for ei in rng.integers(len(es),size=n):
        e = es[ei]
        if dynamics:
            idx = e.focused if len(e.focused) and rng.random()<.5 else e.valid
            yield e, int(rng.choice(idx))
        else:
            yield e, int(rng.integers(len(e.y)))


def normalizers(model, es):
    model.encoder.norm.set_stats(stats(e.x for e in es))
    if isinstance(model,Dynamics):
        model.phys.set_stats(stats(e.episode.physical()[None] for e in es))
        model.action.set_stats(stats(packed(u)[None] for e in es for u in e.episode.a["executed"][e.valid]))
        model.target.set_stats(stats(np.stack((e.delta[e.valid], np.ones_like(e.delta[e.valid]),np.ones_like(e.delta[e.valid])),1) for e in es))


def rng_state(rng):
    return dict(numpy=rng.bit_generator.state,torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(rng,state):
    rng.bit_generator.state=state["numpy"]
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def actor_eval(model, episodes, device, variant, batch):
    errors, phase, target, predictions = [], [], [], []
    model.eval()
    with torch.no_grad():
        for ep in episodes:
            for j in range(0,len(ep.y),batch):
                ids = np.arange(j,min(len(ep.y),j+batch))
                b = gather([ep.item(i,variant) for i in ids],device)
                pred,_ = model.sample(b,deterministic=True)
                p = pred.cpu().numpy()
                y = ep.y[ids]
                errors.append(p-y)
                target.append(y)
                phase.append(ep.episode.a["phase_id"][ids])
                predictions.append(p)
    if not errors:
        return None
    e,y,p=np.concatenate(errors),np.concatenate(target),np.concatenate(predictions)
    stage=np.concatenate(phase)
    def metric(mask):
        return {"n":int(mask.sum()),"mae_normalized":float(abs(e[mask]).mean()),
                "zero_residual_mae_normalized":float(abs(y[mask]).mean()),
                "saturation_fraction":float((abs(p[mask])>=.98).mean())} if mask.any() else {"n":0}
    zero=np.max(abs(y),axis=1)<1e-6
    return dict(overall=metric(np.ones(len(e),bool)), by_phase={i:metric(stage==i) for i in range(1,5)},
                correct_nominal=metric(zero), small_nonzero=metric(~zero & (np.max(abs(y),axis=1)<.35)),
                large=metric(np.max(abs(y),axis=1)>=.35),
                positive_mae=float(abs(e[y>1e-6]).mean()) if (y>1e-6).any() else None,
                negative_mae=float(abs(e[y< -1e-6]).mean()) if (y< -1e-6).any() else None,
                label_outside_bounds=float((abs(y)>1.000001).mean()))


def summarize_delta(error):
    if not len(error):
        return {"n":0}
    def metric(z):
        return [float(np.mean(z)),float(np.quantile(z,.95))]
    return dict(n=len(error),joint_abs_rad_mean_p95=metric(abs(error[:,:43]).mean(axis=1)),
                object_dist_m_mean_p95=metric(np.linalg.norm(error[:,86:89],axis=1)),
                object_angle_rad_mean_p95=metric(np.linalg.norm(error[:,89:92],axis=1)),
                joint_velocity_rad_s_mean_p95=metric(abs(error[:,43:86]).mean(axis=1)),
                object_linear_velocity_m_s_mean_p95=metric(np.linalg.norm(error[:,92:95],axis=1)))


def baseline_delta(a,ids,h):
    return np.c_[a["qd"][ids]*h, np.zeros((len(ids),43)), a["twist"][ids,:3]*h,
                rotation(a["object"][ids,3:]).inv().apply(a["twist"][ids,3:])*h, np.zeros((len(ids),6))]


def ensemble(models,b):
    pred = [m.predict(b) for m in models]
    mu,var,event = map(torch.stack,zip(*pred))
    return mu.mean(0),var.mean(0),mu.var(0,unbiased=False),event.mean(0)


def _evolve(full,de):
    q,qd,p,v = full
    return q+de[:,:43],qd+de[:,43:86],np.c_[p[:,:3]+de[:,86:89],
             wxyz(rotation(p[:,3:])*rotation(wxyz_from_rotvec(de[:,89:92])))],v+de[:,92:98]


def wxyz_from_rotvec(v):
    from scipy.spatial.transform import Rotation
    return wxyz(Rotation.from_rotvec(v))


def errors_multistep(models,ep,ids,steps,device):
    a = ep.episode.a
    batch = [ep.item(i) for i in ids]
    total = tuple(a[k][ids] for k in ("q","qd","object","twist"))
    full = total
    for j in range(steps):
        b = gather(batch,device)
        pred,_,_,_ = ensemble(models,b)
        full = _evolve(full,pred.cpu().numpy())
        if j+1==steps:
            break
        for k,c in enumerate(batch):
            ob = dict(zip(("q","qd","object","twist"),[v[k] for v in full]))
            prev = c["action"]
            c["action"] = a["executed"][ids[k]+j+1]
            new = dyn_frame(ob,prev)
            n = c["length"]
            h = c["history"]
            if n == len(h):
                h[:-1] = h[1:]
                h[-1] = new
            else:
                h[n] = new
                c["length"] += 1
    truth = tuple(a[k][ids+steps] for k in ("q","qd","object","twist"))
    error = state_delta(*truth,*full)
    actual = state_delta(*total,*truth)
    return error,-actual,baseline_delta(a,ids,.02*steps)-actual


def _event_metrics(p,y,m):
    p,y = p[m],y[m]
    if not len(p):
        return {"n":0}
    order = np.argsort(-p,kind="stable")
    ranked = y[order]
    return dict(n=len(y),positives=int(y.sum()),brier=float(np.mean((p-y)**2)),
                average_precision=float(np.sum((np.cumsum(ranked)/np.arange(1,len(y)+1))*ranked)/ranked.sum()) if ranked.any() else None)


def dyn_eval(models,episodes,device,batch):
    from .contract import EVENTS
    for m in models:
        m.eval()
    metrics = {}
    with torch.no_grad():
        for h in (1,5,10):
            errors,stages,positions=[],[],[]
            for e in episodes:
                a=e.episode.a
                v=np.r_[0,np.cumsum(~a["actual_20ms"])]
                ids=np.arange(len(v)-h)
                ids=ids[(v[ids+h]-v[ids])==0]
                if h>1:
                    ids=ids[::5]
                for j in range(0,len(ids),batch):
                    chunk=ids[j:j+batch]
                    errors.append(errors_multistep(models,e,chunk,h,device))
                    stages.append(a["phase_id"][chunk])
            if errors:
                stage=np.concatenate(stages)
                for k,x in zip(("ensemble","hold","constant_velocity"),zip(*errors)):
                    er=np.concatenate(x)
                    assert len(stage)==len(er)
                    positions.append((k,dict(overall=summarize_delta(er),
                        by_phase={i:summarize_delta(er[stage==i]) for i in range(1,5)})))
            metrics[h*20]=dict(positions)
        pp,yy,mm,spread,coverage=[],[],[],[],[]
        for e in episodes:
            for j in range(0,len(e.valid),batch):
                b=gather([e.item(i) for i in e.valid[j:j+batch]],device)
                p,v,d,pr=ensemble(models,b)
                pp.append(pr.cpu().numpy())
                yy.append(b["events"].cpu().numpy())
                mm.append(b["events_valid"].cpu().numpy())
                spread.append(torch.stack([v.sqrt().mean(),d.sqrt().mean()]).cpu().numpy())
                coverage.append(((p-b["delta"]).abs()<=1.645*(v+d).sqrt()).cpu().numpy())
        if pp:
            p,y,m=map(np.concatenate,(pp,yy,mm))
            metrics["events"]={name:_event_metrics(p[:,i],y[:,i],m[:,i]) for i,name in enumerate(EVENTS)}
            metrics["predictive_90_interval_coverage"] = float(np.concatenate(coverage).mean())
            metrics["member_std_and_disagreement_over_channels_native_units"]=np.mean(spread,axis=0)
    return metrics


def train(args):
    out,data=args.out.resolve(),args.data.resolve()
    out.mkdir(parents=True,exist_ok=True)
    cfg=config(args.config)
    device=torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # dataset programs create sorted immutable source IDs before perturbations.
    table=records(data,cfg["history"],args.pilot)
    sonic=Sonic(args.sonic.resolve() if args.sonic else data/"sonic_summary.json",cfg["history"])
    ae={"mixed":{sp:[] for sp in ("train","val","test")}}
    if args.ablations:
        for mode in ("zero_ablation","fixed"):
            ae[mode]={sp:[] for sp in ae["mixed"]}
    de={s:[] for s in ae["mixed"]}
    for ep,r in table:
        if r["native_accepted"] and (r["success"] or args.pilot):
            for k in ae:
                ae[k][ep.split].append(ep.actor(k))
        d=ep.dynamic()
        if d:
            de[ep.split].append(d)
    if not ae["mixed"]["train"] or not de["train"]:
        raise ValueError("Pretraining needs successful residual demonstrations and actual train transitions")
    if not args.pilot and (any(not ae["mixed"][s] for s in ae["mixed"]) or any(not de[s] for s in de)):
        raise ValueError("Pretraining requires held-out residual and actual dynamics episodes")
    identity=dict(schema=SCHEMA,arch=ARCH,seed=args.seed,device=args.device,batch=args.batch,steps=args.steps,
                  aux_steps=args.aux_steps,aux_mix=args.aux_mix,history=cfg["history"],lr=args.lr,pilot=args.pilot,
                  code={f.name:sha256(f) for f in [Path(__file__),Path(__file__).with_name("models.py"),Path(__file__).with_name("features.py")]},
                  datasets=[(str(ep.file),sha256(ep.file)) for ep,_ in table],ablations=args.ablations,
                  sonic_index_sha256=sha256(args.sonic or data/"sonic_summary.json"))
    cp=out/"latest.pt"
    prev=torch.load(cp,map_location="cpu",weights_only=False) if cp.exists() else None
    if prev and (not args.resume or prev["identity"]!=identity):
        raise ValueError("Existing checkpoint needs --resume with unchanged inputs and budget")
    if not prev:
        write(out/"config.json",identity)
    rng=np.random.default_rng(args.seed)
    actors={"mixed":Actor().to(device)}
    if args.ablations:
        for name in ("nominal","zero_ablation","fixed"):
            actors[name]=Actor().to(device)
    ensemble_models=[Dynamics().to(device) for _ in range(3)]
    optim=[torch.optim.AdamW(m.parameters(),lr=args.lr) for m in (*actors.values(),*ensemble_models)]
    boot=[]
    groups=sorted({e.episode.identity["source_group_id"] for e in de["train"]})
    for _ in range(3):
        boot.append([e for g in rng.choice(groups,size=len(groups),replace=True) for e in de["train"]
                     if e.episode.identity["source_group_id"]==g])
    for name,model in actors.items():
        normalizers(model,ae["mixed" if name=="nominal" else name]["train"])
    for m in ensemble_models:
        normalizers(m,de["train"])
    start=0
    if prev:
        start=prev["step"]
        for model,weights in zip(actors.values(),[prev["actor"],*prev["actor_controls"].values()]):
            model.load_state_dict(weights)
        for model,weights in zip(ensemble_models,prev["dynamics"]):
            model.load_state_dict(weights)
        for op,weights in zip(optim,prev["optimizers"]):
            op.load_state_dict(weights)
        restore_rng(rng,prev["rng"])
    wall=time.monotonic()
    train_steps=args.aux_steps+args.steps
    for i in range(start,train_steps):
        losses=[]
        b_aux_a=sonic.batch(rng,min(args.batch,32),ACTOR_SIZE,device)
        b_aux_d=sonic.batch(rng,min(args.batch,32),DYN_SIZE,device)
        for j,(name,model) in enumerate([*actors.items(), *((str(i),m) for i,m in enumerate(ensemble_models))]):
            dyn=isinstance(model,Dynamics)
            aux=b_aux_d if dyn else b_aux_a
            loss=F.smooth_l1_loss(model.track(aux),aux["label"])
            if i>=args.aux_steps:
                if dyn:
                    data_batch=gather([e.item(k) for e,k in choose(boot[j-len(actors)],rng,args.batch,True)],device)
                    loss=args.aux_mix*loss+model.loss(data_batch)
                else:
                    es=ae["mixed" if name=="nominal" else name]["train"]
                    b=gather([e.item(k,"nominal" if name=="nominal" else "full") for e,k in choose(es,rng,args.batch)],device)
                    loss=args.aux_mix*loss+F.smooth_l1_loss(model.sample(b,True)[0],b["label"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite pretraining objective {name} at step {i}")
            op=optim[j]
            op.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            op.step()
            losses.append(float(loss.detach()))
        if (i+1)%args.save_every==0 or i+1==train_steps:
            checkpoint=dict(arch=ARCH,identity=identity,step=i+1,actor=actors["mixed"].state_dict(),
                            actor_controls={n:m.state_dict() for n,m in actors.items() if n!="mixed"},
                            dynamics=[m.state_dict() for m in ensemble_models],optimizers=[opt.state_dict() for opt in optim],
                            rng=rng_state(rng),joint_names=table[0][0].a["joint_names"].tolist())
            save_checkpoint(cp,checkpoint)
            with (out/"metrics.jsonl").open("a") as f:
                print(json.dumps(dict(step=i+1,actor=losses[:len(actors)],dynamics=losses[len(actors):],
                                      wall_seconds=time.monotonic()-wall)),file=f)
            print(f"step={i+1}/{train_steps} loss={np.round(losses,4).tolist()}",flush=True)
        if args.stop_after and i+1>=args.stop_after:
            return
    report=dict(train_updates=train_steps,actual_dynamics={s:sum(len(e.valid) for e in z) for s,z in de.items()},
                dynamics_episodes={s:len(z) for s,z in de.items()},residual_episodes={s:len(z) for s,z in ae["mixed"].items()},
                baselines=Counter(e.mode for e in ae["mixed"]["train"]),device=args.device)
    for s in ("val","test"):
        report[s]=dict(actor={}, dynamics=dyn_eval(ensemble_models,de[s],device,args.batch))
        for name,m in actors.items():
            mode="mixed" if name=="nominal" else name
            report[s]["actor"][name]=actor_eval(m,ae["mixed"][s],device,"nominal" if name=="nominal" else "full",args.batch)
            if name in ("fixed","zero_ablation"):
                report[s]["actor"][name+"_own_distribution"]=actor_eval(m,ae[mode][s],device,"full",args.batch)
        b=sonic.batch(np.random.default_rng(args.seed+71),128,ACTOR_SIZE,device,s)
        with torch.no_grad():
            report[s]["sonic_aux_mae_rad"]=float(.05*(actors["mixed"].track(b)-b["label"]).abs().mean())
    write(out/"evaluation.json",report)
    print(f"Complete: {out/'evaluation.json'}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--sonic",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--config",type=Path,default=ROOT/"configs/experiment.json")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--steps",type=int,required=True)
    p.add_argument("--aux-steps",type=int,default=100)
    p.add_argument("--aux-mix",type=float,default=.1)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--ablations",action="store_true")
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--stop-after",type=int)
    args=p.parse_args()
    train(args)


if __name__=="__main__":
    main()

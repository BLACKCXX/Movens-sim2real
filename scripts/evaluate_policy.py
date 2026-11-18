"""Evaluate a frozen base and optional residual on predeclared source groups."""
import argparse
import fcntl
from pathlib import Path
import signal

from movens_sim2real.contract import config, sha256, write
from movens_sim2real.data import split_apple
from movens_sim2real.deployment import factory
from movens_sim2real.learn import online_dynamics_summary, stat_rows
from movens_sim2real.models import load_models
from movens_sim2real.rollout import episode_rollout, policy_load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--episode",type=int,nargs="+",required=True)
    p.add_argument("--base-adapter",type=Path)
    p.add_argument("--small",type=Path)
    p.add_argument("--modes",nargs="+",choices=("clean","physics","noisy","deployment"),default=["clean"])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    cfg=config()
    splits=split_apple(cfg)
    if len(set(a.episode))!=len(a.episode) or any(splits[i]!="test" for i in a.episode):
        raise ValueError("Formal evaluation requires distinct test source groups")
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/".run.lock").open("w") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop=[]
        for s in (signal.SIGTERM,signal.SIGINT):
            signal.signal(s,lambda *_:stop.append(True))
        model=policy_load(cfg,a.device,a.base_adapter)
        actor,dynamics,small=None,[],None
        if a.small:
            actor,dynamics,_=load_models(a.small,a.device)
            small=dict(path=a.small.resolve(),sha256=sha256(a.small))
        result={}
        for mode in a.modes:
            rows=[]
            for ep in a.episode:
                if stop:
                    return
                r=episode_rollout(cfg,ep,a.out/mode/f"apple_{ep:06d}",model,actor,dynamics,
                    small_identity=small,seed=a.seed,side=True,adapter_factory=factory(mode) if mode!="clean" else None,
                    stop=lambda:bool(stop))
                if r is None:
                    return
                rows.append(r)
            result[mode]=dict(stat_rows(rows),d0=online_dynamics_summary(rows),
                              episodes=[r["identity"]["episode"] for r in rows])
            write(a.out/"evaluation.json",result)


if __name__=="__main__":
    main()

"""Audit the fixed M5 training receipt and paired S0/S1 evaluations."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from compare_rollouts import compare, observation_evidence
from movens_sim2real.contract import read, sha256, write
from movens_sim2real.data import array_dict
from movens_sim2real.export import audit


MODES=("clean","physics","noisy","deployment")
EPISODES=(7,16)
CURRICULUM=((1,"physics"),(8,"noisy"),(2,"deployment"),(4,"deployment"))


def inputs(root,expected):
    if (root/"exit").read_text().strip()!="0":
        raise ValueError(f"{root}: missing a successful worker exit")
    results=read(root/"evaluation.json")
    if expected and (set(results)!=set(expected) or any(v["episodes"]!=list(EPISODES) or v["complete"]!=2
                    or v["engineering_errors"] or v["control_ticks"]!=1353 for v in results.values())):
        raise ValueError(f"{root}: incomplete frozen evaluation")
    return results


def paired_conditions(paths):
    evidence=[]
    for k,ep in enumerate(EPISODES):
        initial={m:array_dict(paths[m][k].parent/"initial.npz")["integration"] for m in MODES}
        a={m:array_dict(paths[m][k].parent/"trajectory.npz") for m in MODES}
        if (not np.array_equal(initial["clean"],initial["noisy"])
                or not np.array_equal(initial["physics"],initial["deployment"])
                or np.array_equal(initial["clean"],initial["physics"])
                or not np.array_equal(a["clean"]["physical"],a["noisy"]["physical"])
                or not np.array_equal(a["physics"]["physical"],a["deployment"]["physical"])
                or a["clean"]["physical"][0]==a["physics"]["physical"][0]):
            raise ValueError(f"Episode {ep}: physical change and measurement change were not isolated")
        d=a["deployment"]
        known=d["observed_metadata"][:,0].astype(bool)
        if (not known.any() or known.all() or d["observed_point_valid"][~known].any()
                or d["observed_physical"][:,1,:16].any() or d["observed_physical"][:,1,26:].any()
                or d["actor_frames"][:,1,89:95].any() or d["dynamics_frames"][:,1,89:95].any()
                or d["actor_frames"][:,1,43:86].all()):
            raise ValueError(f"Episode {ep}: missing real optical estimates or masked deployment features")
        evidence.append(dict(episode=ep,nominal_mass_kg=a["clean"]["physical"][0],
            randomized_mass_kg=a["physics"]["physical"][0],deployment_visible=int(known.sum()),
            deployment_missing=int((~known).sum())))
    return evidence


def training_evidence(root,cfg):
    rows=[]
    for j,(ep,mode) in enumerate(CURRICULUM):
        p=root/"train"/f"{j:03d}_apple_{ep:06d}"/"summary.json"
        row,a,_=audit(p)
        i=row["identity"]
        if not row["complete"] or i["split"]!="train" or i["adapter"]["mode"]!=mode or i["episode"]!=ep:
            raise ValueError(f"{p}: unexpected source or physical/observation condition in the curriculum")
        rows.append(dict(episode=ep,mode=mode,steps=row["decision_steps"],success=row["success"],
            failure=row["failure"],export_eligible=row["success_export_eligible"],inputs=observation_evidence(a),
            summary=str(p.resolve()),summary_sha256=sha256(p)))
    for r in cfg["execution"]["clean_replay"]:
        row=read(r["path"])
        if (sha256(r["path"])!=r["summary_sha256"] or sha256(row["path"])!=r["trajectory_sha256"]
                or row["identity"]["split"]!="train" or row["identity"]["adapter"] is not None):
            raise ValueError("Clean S0 replay source or split changed after training")
    return rows


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    runs=args.runs
    train=inputs(runs/"m5_s1",None)
    cfg=read(runs/"m5_s1/config.json")
    clean=runs/"m4_s0_pretrained"
    if (train["env_steps"]!=3428 or not 0<train["updates"]<=480 or train["train"]["complete"]!=4
            or train["train"]["engineering_errors"] or cfg["execution"]["pretrained"]!=sha256(clean/"final.pt")):
        raise ValueError("Training did not complete its declared budgets, independent of success rate")
    train_rows=training_evidence(runs/"m5_s1",cfg)
    changes=inputs(runs/"m5_eval_s0",MODES[1:])
    baseline={"clean":read(clean/"evaluation.json")["test_splits"]["clean"],**changes}
    s1_hash=sha256(runs/"m5_s1/final.pt")
    s0_hash=cfg["execution"]["pretrained"]
    s0_clean=dict(path=str((clean/"final.pt").resolve()),size=(clean/"final.pt").stat().st_size)
    results,paths={},[{},{}]
    for mode in MODES:
        root=runs/"m5_eval_s1"/mode
        s1=inputs(root,[mode])[mode]
        s0=clean/"eval" if mode=="clean" else runs/"m5_eval_s0"
        for dest,base in zip(paths,(s0,root)):
            dest[mode]=[base/mode/f"apple_{ep:06d}"/"summary.json" for ep in EPISODES]
        for index,expected in ((0,s0_hash),(1,s1_hash)):
            for receipt in paths[index][mode]:
                i=read(receipt)["identity"]
                small=i["small"] or {}
                same_weights=small==s0_clean if index==0 and mode=="clean" else small.get("sha256")==expected
                if i["mode"]!="D0" or not i["d0_predictions"] or not i["residual_enabled"] or not same_weights:
                    raise ValueError(f"{receipt}: wrong frozen residual/D0 checkpoint or execution mode")
        pair=compare([("s0",paths[0][mode]),("s1",paths[1][mode])])
        pair.update(s0_d0=baseline[mode]["d0"],s1_d0=s1["d0"])
        results[mode]=pair
    write(args.out,dict(training=train,training_trajectories=train_rows,conditions=results,s0_sha256=s0_hash,s1_sha256=s1_hash,
        physical_vs_perception={"s0":paired_conditions(paths[0]),
        "s1":paired_conditions(paths[1])},joint_timing="50 Hz; 80 original physical steps per decision",dynamics_mode="D0"))


if __name__=="__main__":
    main()

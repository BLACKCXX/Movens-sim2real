"""Compare complete, paired rollouts without silently dropping failures."""
import argparse
from pathlib import Path

import numpy as np

from movens_sim2real.contract import read, sha256, write
from movens_sim2real.data import array_dict
from movens_sim2real.export import audit
from movens_sim2real.learn import stat_rows


def observation_evidence(a):
    m=a["observed_metadata"]
    known=m[:,0].astype(bool)
    edges=np.diff(np.r_[False,~known,False].astype(int))
    gaps=np.flatnonzero(edges<0)-np.flatnonzero(edges>0)
    phase_valid=m[:,5].astype(bool)&a["phase_valid"]
    age=None  # Original clean S0 logs predate the separate encoder/vision capture stamps.
    if "observed_estimate_q_capture_s" in a:
        age=a["t"]-a["observed_estimate_q_capture_s"]
        if np.any(age<-1e-8) or np.any(a["observed_estimate_object_capture_s"][known]>a["t"][known]+1e-8):
            raise ValueError("Observation was captured after the recorded decision")
    error=np.linalg.norm(a["observed_estimate_object"][known,:3]-a["object"][known,:3],axis=1)
    return dict(boundaries=len(m),object_visible=int(known.sum()),
        object_missing_segments=len(gaps),longest_object_gap_ticks=int(gaps.max()) if len(gaps) else 0,
        visible_geometry_frames=int(a["observed_point_valid"].any(axis=1).sum()),
        known_encoder_rate_frames=int(a["actor_frames"][:,1,43:86].all(axis=1).sum()),
        known_orientation_frames=int(a["actor_frames"][:,1,89:95].all(axis=1).sum()),
        phase_comparable=int(phase_valid.sum()),phase_disagreements=int(np.sum((m[:,4]!=a["phase_id"])&phase_valid)),
        object_center_error_p95_m=float(np.quantile(error,.95)) if len(error) else None,
        max_observation_age_s=float(m[:,2].max()),max_encoder_age_s=float(age.max()) if age is not None else None)


def compare(arms, out=None):
    reports={}
    for name, receipts in arms:
        if name in reports:
            raise ValueError(f"Repeated arm name {name}")
        rows={}
        for p in receipts:
            row=read(p)
            i=row["identity"]
            ep=i["episode"]
            if (ep in rows or i["split"]!="test" or i["training"] or row["engineering_error"]
                    or not row["complete"] or not row["reached_protocol_initial"]):
                raise ValueError(f"{p}: incomplete, invalid or non-test paired rollout")
            rows[ep]=(p,row)
        reports[name]=rows
    control=next(iter(reports.values()))
    ep_ids=sorted(control)
    initials={ep:array_dict(control[ep][0].parent/"initial.npz") for ep in ep_ids}
    matched=("task","seed","rules","rule_version","dt","physics_dt","side","trial","asset","adapter",
             "source_group_id","reset_id","phase_mode","history_length")
    models=("policy","small","mode","residual_enabled","d0_predictions")
    out_rows={}
    for name, rows in reports.items():
        if set(rows)!=set(ep_ids):
            raise ValueError(f"{name}: changed the fixed held-out episode set")
        model={k:rows[ep_ids[0]][1]["identity"][k] for k in models}
        paired=[]
        for ep in ep_ids:
            p,x=rows[ep]
            y=control[ep][1]
            if any(x["identity"][k]!=v for k,v in model.items()):
                raise ValueError(f"{p}: changed the frozen controller within a comparison arm")
            if (any(x["identity"][k]!=y["identity"][k] for k in matched)
                    or x["decision_steps"]!=y["decision_steps"]):
                raise ValueError(f"{p}: fixed paired conditions or interaction horizon changed")
            _,a,_=audit(p)
            current=array_dict(p.parent/"initial.npz")
            if any(not np.array_equal(current[k],initials[ep][k]) for k in ("integration","qpos","qvel","actuator")):
                raise ValueError(f"{p}: paired execution changed the initial integration state")
            paired.append(dict(episode=ep,success=x["success"],failure=x["failure"],
                stable_release_s=x["stable_release_s"],success_delta=int(x["success"])-int(y["success"]),
                inputs=observation_evidence(a),receipt=str(p.resolve()),receipt_sha256=sha256(p)))
        out_rows[name]=dict(model=model,stats=stat_rows([v[1] for v in rows.values()]),pairs=paired,
            success_count_delta=sum(r["success_delta"] for r in paired))
    result=dict(reference=arms[0][0],test_episodes=ep_ids,arms=out_rows)
    if out is not None:
        write(out,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",nargs="+",action="append",required=True,metavar="NAME_OR_RECEIPT")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if len(a.arm)<2 or any(len(v)<2 for v in a.arm):
        p.error("Provide at least two --arm NAME SUMMARY [SUMMARY ...] groups")
    compare([(v[0],[Path(s) for s in v[1:]]) for v in a.arm],a.out)

"""Continue the Movens Body/Hand adapters on a fixed mixture of old and successful data."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors import safe_open

from movens_interact.backbone import _ActionExpert
from movens_interact.hand import HandExpert
from movens_interact.kinematics import G1Kinematics
from movens_interact.model import _normalizer
from training.image_sources import reconcile_jsonl
from training.mixed_batch import MixedBatchState, accumulated_backward
from training.paired_data import SimWindows
from training.train import evaluate_precise, precise_batch

from .contract import ROOT, config, read, sha256, write
from .data import split_apple
from .models import save_checkpoint
from .pretrain import restore_rng, rng_state
from .rollout import file_identity


def dataset(index,cache,body_norm,hand_norm,split,*,verify_source=False):
    manifest=read(cache/"manifest.json")
    if (not manifest["complete"] or manifest["teacher_index_sha256"]!=sha256(index)
            or not manifest["prompt"] or split not in manifest["splits"]):
        raise ValueError(f"{cache}: incomplete or incompatible original Qwen cache")
    loader=SimWindows
    if read(index).get("domain")=="recorded_hardware":
        from .real_export import HardwareWindows
        loader=HardwareWindows
    return loader(index,cache,body_norm,hand_norm,split,stride=10,verify_hashes=verify_source),manifest


def load_experts(x,base,device):
    body_modules=x["body_trainable_modules"]
    prefixes=tuple(n+"." for n in body_modules)
    with safe_open(str(base),framework="pt",device="cpu") as f:
        weights={k.removeprefix("action_header."):f.get_tensor(k)
                 for k in f.keys() if k.startswith("action_header.")}
    with torch.device("meta"):
        body=_ActionExpert(x["config"]["body_config"])
    body.load_state_dict(weights,strict=True,assign=True)
    trainable=[name for name,p in body.named_parameters() if p.requires_grad and name.startswith(prefixes)]
    body.to(device=device,dtype=torch.bfloat16)
    for n in body_modules:
        body.get_submodule(n).float()
    if set(x["body_delta_state_dict"])!={k for k in body.state_dict() if k.startswith(prefixes)}:
        raise ValueError("Parent Body40 adapter state differs from the declared modules")
    body.load_state_dict(x["body_delta_state_dict"],strict=False)
    body.requires_grad_(False).eval()
    for name in trainable:
        body.get_parameter(name).requires_grad_(True)
    hand=HandExpert(**x["config"]["hand_config"]).to(device).float()
    hand.load_state_dict(x["hand_state_dict"],strict=True)
    return body,hand,prefixes


def train(a):
    cfg=config(a.config)
    out=a.out.resolve()
    source=cfg["checkpoint"]
    new_index=a.new_index.resolve()
    new_cache=a.new_context.resolve()
    checks=read(new_index.parent/"verification.json")
    if max(checks["body29_roundtrip_max_rad"],checks["hand14_roundtrip_max_rad"])>1e-6:
        raise ValueError("New demonstrations failed the codec check")
    x=torch.load(source,map_location="cpu",weights_only=False,mmap=True)
    provenance=x["data_provenance"]
    old_index=Path(provenance["teacher_index"])
    old_cache=Path(provenance["context_manifest"]).parent
    if (provenance["teacher_index_sha256"]!=sha256(old_index)
            or provenance["context_manifest_sha256"]!=sha256(old_cache/"manifest.json")):
        raise ValueError("Parent paired index or cache changed")
    index=read(new_index)
    previous=read(old_index)
    if any(index[k]!=previous[k] for k in ("kinematics_sha256","body_joint_names","hand_joint_names","data_revision")):
        raise ValueError("New and old source kinematics or mappings differ")
    if not index["episodes"] or any(r["split"]!="train" for r in index["episodes"]):
        raise ValueError("New demonstrations overlap held-out source groups")
    domain=index.get("domain","mujoco")
    if domain=="recorded_hardware":
        from .real_export import split_hardware_sources
        records=[]
        for row in index["episodes"]:
            path=Path(row["source_trial"])
            m=read(path)
            if sha256(path)!=row["source_trial_sha256"] or any(m[k]!=row[k] for k in ("trial_id","split","source_group_id")):
                raise ValueError(f"{path}: hardware trial or source split changed after export")
            records.append(m)
        split_hardware_sources(records,cfg)
    elif domain=="mujoco":
        groups=split_apple(cfg)
        if any(groups.get(r["original_episode_index"])!="train" or
               r["source_group_id"]!=f"apple-d89c126a-{r['original_episode_index']:06d}" for r in index["episodes"]):
            raise ValueError("New demonstrations overlap held-out source groups")
    else:
        raise ValueError(f"Unknown successful data domain {domain}")
    device=torch.device(a.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction,device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    bnorm,hnorm=_normalizer(x["body_normalization"]),_normalizer(x["hand_normalization"])
    old,old_manifest=dataset(old_index,old_cache,bnorm,hnorm,"train")
    val,_=dataset(old_index,old_cache,bnorm,hnorm,"val")
    new,new_manifest=dataset(new_index,new_cache,bnorm,hnorm,"train",verify_source=True)
    if (new_manifest["prompt"]!=cfg["task"]
            or any(old_manifest[k]!=new_manifest[k] for k in ("checkpoint_sha256","qwen_configuration_sha256","body_config_sha256"))
            or any(old_manifest["preprocessing"][k]!=new_manifest["preprocessing"][k]
                   for k in ("video_sample_fps","video_size_pixels","temporal_patching","context_dtype"))):
        raise ValueError("Parent and new frozen visual feature protocols differ")
    effective=a.batch
    train_cfg=deepcopy(x["config"])
    train_cfg["training"].update(seed=a.seed,batch_size=a.microbatch,effective_batch_size=effective,
        body_lr=a.body_lr,hand_lr=a.hand_lr,verification=False,eval_batches=a.eval_batches,
        domain_weighting="none",mix_mode="fixed",device=str(device),output=str(out),gpu_memory_fraction=a.gpu_memory_fraction)
    args=SimpleNamespace(**train_cfg["training"])
    policy=train_cfg["hand_action_policy"]
    args._hand_policy=policy
    # RGB caches use the same frozen backbone. Exact old/new sample quotas are
    # fixed outside the original effective-batch loss, which is reused intact.
    mix=MixedBatchState(args,None)
    identity=dict(parent=file_identity(source),old_index=sha256(old_index),old_cache=sha256(old_cache/"manifest.json"),
        new_index=sha256(new_index),new_cache=sha256(new_cache/"manifest.json"),
        steps=a.steps,seed=a.seed,batch=effective,microbatch=a.microbatch,new_per_batch=a.new_per_batch,
        body_lr=a.body_lr,hand_lr=a.hand_lr,device=str(device),eval_batches=a.eval_batches,gpu_memory_fraction=a.gpu_memory_fraction,
        objective="training.mixed_batch.accumulated_backward; original flow, FK and physical losses",
        trainable="original Body adapter modules and Hand expert; new bounded Adam state",
        code=sha256(Path(__file__)))
    path=out/"checkpoint_last.pt"
    if (out/"config.json").exists():
        if not a.resume or read(out/"config.json")!=identity:
            raise ValueError("Refit resume requires unchanged configuration and dataset identities")
    else:
        write(out/"config.json",identity)
    body,hand,prefixes=load_experts(x,x["base_checkpoint"],device)
    hand.train()
    fk=G1Kinematics(previous["scene_xml"],previous["body_joint_names"]).to(device=device,dtype=torch.float32)
    import mujoco
    robot=mujoco.MjModel.from_xml_path(previous["scene_xml"])
    ids=[robot.joint(n).id for n in previous["body_joint_names"]]
    bounds={k:torch.as_tensor(v,device=device,dtype=torch.float32) for k,v in dict(
        body_lo=bnorm.action_min[:29],body_hi=bnorm.action_max[:29],hand_lo=hnorm.action_min,hand_hi=hnorm.action_max,
        limit_lo=robot.jnt_range[ids,0],limit_hi=robot.jnt_range[ids,1]).items()}
    del robot
    weights=torch.ones(40,device=device)/40
    for sl,s in ((slice(29,31),3),(slice(31,37),10),(slice(37,38),15),(slice(38,40),.1)):
        weights[sl]*=s
    wrists=torch.tensor([args.left_wrist_weight,1.],device=device)
    bp=[p for p in body.parameters() if p.requires_grad]
    hp=list(hand.parameters())
    opt=torch.optim.Adam([dict(params=hp,lr=a.hand_lr),dict(params=bp,lr=a.body_lr)])
    start=0
    if path.exists():
        z=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        if z["refit_identity"]!=identity:
            raise ValueError("Checkpoint provenance changed")
        start=z["refit_step"]
        body.load_state_dict(z["body_delta_state_dict"],strict=False)
        hand.load_state_dict(z["hand_state_dict"],strict=True)
        opt.load_state_dict(z["optimizer_state_dict"])
        restore_rng(rng,z["rng_state"])
        del z
        reconcile_jsonl(out/"metrics.jsonl",start)
    keys=("format","base_identity","base_checkpoint","base_sha256","body_normalization_sha256",
          "body_normalization","hand_normalization","body_trainable_modules","kinematics_metadata")
    checkpoint={k:x[k] for k in keys}
    parent_step=x["step"]
    del x
    p=dict(provenance,context_manifest=str(new_cache/"manifest.json"),
           context_manifest_sha256=identity["new_cache"],context_prompt=cfg["task"],new_teacher_index=str(new_index),
           new_teacher_index_sha256=identity["new_index"])
    train_cfg.update(data_provenance=p,refit=identity)
    def save(step):
        save_checkpoint(path,dict(checkpoint,step=parent_step+step,refit_step=step,refit_identity=identity,
            config=train_cfg,data_provenance=p,optimizer_has_body_group=True,
            body_delta_state_dict={k:v.detach().cpu() for k,v in body.state_dict().items() if k.startswith(prefixes)},
            hand_state_dict={k:v.detach().cpu() for k,v in hand.state_dict().items()},
            optimizer_state_dict=opt.state_dict(),rng_state=rng_state(rng)))
    stop=[]
    for s in (signal.SIGTERM,signal.SIGINT):
        signal.signal(s,lambda *_:stop.append(True))
    began=time.monotonic()
    domains=np.repeat("sim",effective)
    for step in range(start,a.steps):
        ids_old=rng.integers(len(old),size=effective-a.new_per_batch)
        ids_new=rng.integers(len(new),size=a.new_per_batch)
        batch=[old[int(i)] for i in ids_old]+[new[int(i)] for i in ids_new]
        batch=precise_batch([batch[j] for j in rng.permutation(effective)],device)
        opt.zero_grad(set_to_none=True)
        result=accumulated_backward(body,hand,fk,batch,args,bounds,policy,weights,wrists,step,True,domains,mix)
        hgrad,bgrad=(float(torch.nn.utils.clip_grad_norm_(p,1.,error_if_nonfinite=True)) for p in (hp,bp))
        if not hgrad or not bgrad:
            raise FloatingPointError("Original Body or Hand objective produced no effective gradient")
        opt.step()
        if not all(bool(torch.isfinite(v).all()) for v in (*bp,*hp)):
            raise FloatingPointError("Nonfinite adapter update; prior checkpoint retained")
        completed=step+1
        metrics=dict(step=completed,loss=result["loss"],hand_grad_norm=hgrad,body_grad_norm=bgrad,
                     old_windows=len(ids_old),new_windows=len(ids_new),parts=result["components"],
                     wall_s=time.monotonic()-began)
        with (out/"metrics.jsonl").open("a") as f:
            print(json.dumps(metrics,allow_nan=False),file=f)
        print(metrics,flush=True)
        pause=a.stop_after is not None and completed>=a.stop_after
        if completed%a.save_every==0 or completed==a.steps or stop or pause:
            save(completed)
        if stop or pause:
            return
    if not path.exists():
        raise ValueError("No Movens adapter was saved")
    report=evaluate_precise(body,hand,fk,val,args,bounds,"retained_validation")
    write(out/"evaluation.json",dict(updates=a.steps,checkpoint=file_identity(path),same_old_val_split=report))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/experiment.json")
    p.add_argument("--new-index",type=Path,required=True)
    p.add_argument("--new-context",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--microbatch",type=int,default=2)
    p.add_argument("--new-per-batch",type=int,default=2)
    p.add_argument("--body-lr",type=float,default=1e-5)
    p.add_argument("--hand-lr",type=float,default=4e-5)
    p.add_argument("--eval-batches",type=int,default=4)
    p.add_argument("--save-every",type=int,default=20)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--gpu-memory-fraction",type=float,default=.9)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--stop-after",type=int,help="Commit and pause at the absolute update, preserving the full preset budget")
    args=p.parse_args()
    if (min(args.batch,args.microbatch,args.steps,args.save_every)<=0
            or not 0<=args.new_per_batch<args.batch or args.eval_batches<0 or not args.device.startswith("cuda")
            or not 0<args.gpu_memory_fraction<=1
            or args.stop_after is not None and not 0<args.stop_after<args.steps):
        p.error("Invalid finite refit budget and mixture")
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/".run.lock").open("w") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train(args)


if __name__=="__main__":
    main()

"""Convert independently verified hardware successes without calling them simulated data."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
from training.paired_data import BASE_KEYS, FORMAT, SPARSE_KEYS, SimWindows, robot_model_info, rollout_window_data

from .calibration import Trial
from .contract import read, save_arrays, sha256, split_map
from .data import array_dict, split_apple
from .export import finish_export
from .real_adapt import measured_returns


ROOT_SOURCE="calibrated_fixed_base_hardware"


class HardwareWindows(SimWindows):
    """Same codec, window and Qwen payload as SimWindows with an audited hardware root."""

    def __init__(self,index,cache,body_norm,hand_norm,split,stride=10,verify_hashes=True):
        self.path=Path(index)
        meta=read(index)
        if meta["domain"]!="recorded_hardware" or meta["format"]!=FORMAT or meta["fps"]!=50 or stride!=10:
            raise ValueError("Expected audited fixed-base hardware windows on the shared 50 Hz codec")
        self.body_norm,self.hand_norm=body_norm,hand_norm
        self.context_root=Path(cache) if cache is not None else None
        self.context_lru=OrderedDict()
        self.episodes,self.rows,self.windows=[],[],[]
        context={}
        if cache is not None:
            m=read(Path(cache)/"manifest.json")
            # The upstream cache calls its row-NPZ image reader "sim".
            # Here the verified row and its images still have hardware provenance.
            if (m["format"]!="movens_qwen_sharded_v3" or not m["complete"] or m["teacher_index_sha256"]!=sha256(index)
                    or m.get("image_source","sim")!="sim"):
                raise ValueError("Hardware RGB must use the original frozen sharded cache over exported arrays")
            context={r["episode_index"]:r for r in m["episodes"]}
        for row in meta["episodes"]:
            if row["split"]!=split:
                continue
            if (not row["teacher_success"] or not row["success"] or not row["rgb_ready"]
                    or row["root_source"]!=ROOT_SOURCE or not row["root_calibration_source"]
                    or row["kinematics_sha256"]!=meta["kinematics_sha256"]):
                raise ValueError("Invalid real demonstration or fixed-base calibration")
            if verify_hashes:
                for k in ("trajectory","rgb","scene","source_trial"):
                    path=row["scene_xml"] if k=="scene" else row[k]
                    if sha256(self.path.parent/path)!=row[k+"_sha256"]:
                        raise ValueError(f"Verified hardware data changed: {path}")
            with np.load(self.path.parent/row["trajectory"],allow_pickle=False) as z:
                ep={key:z[key] for key in BASE_KEYS+SPARSE_KEYS}
            ts,ticks=ep["timestamps"],ep["observation_control_indices"]
            n,nobs=len(ts),len(ticks)
            specs=dict(body_obs_abs=(nobs,36),body_ref_abs=(n,36),hand_obs=(nobs,14),hand_ref=(n,14))
            if (nobs<1 or n<2 or not np.allclose(ts,np.arange(n)*.02,rtol=0,atol=1e-8)
                    or any(ep[k].shape!=shape for k,shape in specs.items()) or not all(np.isfinite(v).all() for v in ep.values())
                    or not np.array_equal(ticks,np.arange(0,n-1,5))
                    or not np.allclose(ts[ticks],ep["observation_timestamps"],rtol=0,atol=1e-9)
                    or not np.array_equal(ep["observation_timestamps"],ep["rgb_timestamps"])
                    or np.max(abs(np.r_[ep["body_obs_abs"][:,29:],ep["body_ref_abs"][:,29:]]-ep["body_obs_abs"][0,29:]))>1e-6):
                raise ValueError("Broken measured action grid, RGB pair or fixed hardware root")
            eid=row["episode_index"]
            if cache is not None:
                r=context[eid]
                if not r["complete"] or r["source_sha256"]!=row["rgb_sha256"] or not np.array_equal(r["timestamps"],ep["rgb_timestamps"]):
                    raise ValueError("Hardware RGB cache does not cover the original paired windows")
                ep.update(context_times=np.asarray(r["timestamps"]),context_chunks=r["chunks"])
            self.windows.extend((len(self.episodes),j) for j,tick in enumerate(ticks) if tick%stride==0)
            self.rows.append(row)
            self.episodes.append(dict(ep,episode_index=eid,perturbation_seed=None))
        if not self.windows:
            raise ValueError(f"No accepted hardware windows in {split}")


def split_hardware_sources(records, cfg):
    groups=split_apple(cfg)
    known={f"apple-d89c126a-{ep:06d}":split for ep,split in groups.items()}
    assign={split:{r["source_group_id"] for r in records if r["split"]==split} for split in ("train","val","test")}
    if (len({r["trial_id"] for r in records})!=len(records)
            or any(r["split"] not in assign for r in records)
            or any(known.get(g,split)!=split for g,split in split_map(assign).items())):
        raise ValueError("Duplicate hardware trial ID or conflicting source split")


def paired_hardware(a, root, rgb):
    """Create a nominal lattice for supervised codecs; keep all raw clocks separately."""
    active=np.flatnonzero(a["decision_mask"])
    t=a["t"]
    grid=np.arange(len(t))*.02
    if (not len(active) or not np.array_equal(active,np.arange(len(active))) or active[-1]>=len(t)-1
            or np.max(abs(t-t[0]-grid))>.002
            or not np.allclose(a["applied_end_s"][:-1],a["applied_start_s"][1:],rtol=0,atol=2e-4)):
        raise ValueError("Cannot map discontinuous or drifting measured commands to native trainer ticks")
    k=np.arange(0,len(active),5)
    stamps=rgb["captured_s"]
    pixels=rgb["rgb"]
    if (stamps.ndim!=1 or not len(stamps) or not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0)
            or pixels.shape!=(len(stamps),480,640,3) or pixels.dtype!=np.uint8):
        raise ValueError("Expected timestamped native uint8 RGB at 640x480")
    ix=np.searchsorted(stamps,t[k],side="right")-1
    if (np.any(ix<0) or np.any(t[k]-stamps[ix]>.035) or np.any(np.diff(ix)<=0)
            or not (a["measured_valid"][k,:43]&(a["measured_confidence"][k,:43]>=.6)).all()):
        raise ValueError("No jointly usable, causal RGB and measured encoder states at the trainer boundaries")
    q=a["q"][k]
    d=rollout_window_data(dict(timestamp_s=grid[k],control_timestamp_s=grid[:-1],control_targets_rad=a["executed"],
        body_obs_abs36=np.c_[q[:,:29],np.broadcast_to(root,(len(q),7))],root_pose_wxyz=root[None],
        hand_obs14=q[:,29:],frame_index=np.arange(len(k))),task_end=grid[len(active)])
    d.update(applied_target43=a["executed"][active],applied_start_s=a["applied_start_s"][active],
             applied_end_s=a["applied_end_s"][active],encoder_capture_s=t[k],rgb_capture_s=stamps[ix],native_rgb_indices=ix)
    return d,pixels[ix]


def export_hardware(out,paths,cfg):
    template=read(cfg["teacher"]/"index.json")
    body,hand=template["body_joint_names"],template["hand_joint_names"]
    trials=[Trial(p.resolve(),body+hand) for p in sorted(set(paths))]
    split_hardware_sources([tr.meta for tr in trials],cfg)
    rows,excluded,expected=[],[],{}
    for tr in trials:
        r,a,p=tr.meta,tr.a,tr.path
        if r["split"]!="train" or not r["complete"] or r.get("engineering_error") or not r.get("reached_protocol_initial"):
            excluded.append(dict(path=p,reason="held out, incomplete, invalid or no protocol initial state"))
            continue
        status=measured_returns(tr,cfg,require_replay=False)["outcome"]
        if not status or not status["success"]:
            excluded.append(dict(path=p,reason=status["failure"] if status else "insufficient measured outcome evidence"))
            continue
        root=np.asarray(r["root_xyz_wxyz"],dtype=float)
        xml=(p.parent/r["scene_xml"]).resolve()
        if (root.shape!=(7,) or not np.isfinite(root).all() or abs(np.linalg.norm(root[3:])-1)>1e-5
                or not r["root_calibration_source"] or not r["initial_evidence_source"] or not r["rgb_clock_source"]
                or sha256(xml)!=r["scene_sha256"]
                or robot_model_info(xml,body,hand)["kinematics_sha256"]!=template["kinematics_sha256"]):
            raise ValueError(f"{p}: calibrated fixed-base pose or robot revision is incompatible")
        plans=p.parent/r["plans"]
        rgb=p.parent/r["rgb"]
        d,images=paired_hardware(a,root,array_dict(rgb))
        z=array_dict(plans)
        count=len(d["applied_target43"])
        b,k=np.divmod(np.arange(count),20)
        order=np.arange(b[-1]+1)
        if (not np.array_equal(z["plan_id"],order) or not np.array_equal(z["first_physical_frame"],1+20*order)
                or not np.array_equal(z["prefix_length"],np.where(order==0,0,10))
                or z["body_raw"].shape!=(len(order),30,40) or not np.isfinite(z["body_raw"]).all()
                or not np.isfinite(z["anchor"]).all()
                or not np.allclose(np.c_[z["body_absolute"][b,k,:29],z["hand_absolute"][b,k]],a["nominal"][:count],rtol=0,atol=1e-6)):
            raise ValueError(f"{p}: stored nominal Body40/Hand14 chunk does not match actual control decisions")
        d.update(nominal_body40_chunks=z["body_raw"],nominal_body40_anchor=z["anchor"],nominal_hand14_chunks=z["hand_absolute"],
                 task_text=np.array(r["task"]),source_group_id=np.array(r["source_group_id"]))
        eid=2_000_000+len(rows)
        dp,rp=(Path(folder)/f"episode_{eid:06d}.npz" for folder in ("episodes","rgb"))
        save_arrays(out/dp,**d)
        save_arrays(out/rp,rgb=images,timestamps=d["rgb_timestamps"],frame_indices=d["rgb_source_frame_indices"])
        rows.append(dict(episode_index=eid,source_group_id=r["source_group_id"],trial_id=r["trial_id"],reset_id=a["reset_id"][0],
            trajectory=str(dp),trajectory_sha256=sha256(out/dp),rgb=str(rp),rgb_sha256=sha256(out/rp),scene_xml=str(xml),
            scene_sha256=r["scene_sha256"],kinematics_sha256=template["kinematics_sha256"],split="train",success=True,
            teacher_success=True,rgb_ready=True,root_source=ROOT_SOURCE,root_calibration_source=r["root_calibration_source"],
            frames=len(d["timestamps"]),observation_frames=len(images),source_trial=str(p),source_trial_sha256=sha256(p),
            source_trajectory_sha256=sha256(p.parent/r["trajectory"]),source_plans_sha256=sha256(plans),source_rgb_sha256=sha256(rgb)))
        expected[eid]=d["applied_target43"]
    finish_export(out,template,cfg,rows,excluded,expected,
                  "acknowledged actual Body29; independently calibrated fixed hardware root; original Body40 retained",
                  domain="recorded_hardware",loader=HardwareWindows)

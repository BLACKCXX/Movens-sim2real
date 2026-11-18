"""Calibrated observation and SDK action boundaries for the fixed-base task."""
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import mujoco as mj
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from .contract import GROUPS, rotation, wxyz
from .features import actor_frame, dyn_frame
from .vision import empty_cloud, partial_points, unproject


class Latest:
    """One timestamped result per stream; no sensor or network I/O on read."""
    def __init__(self, max_age):
        if not np.isfinite(max_age) or max_age <= 0:
            raise ValueError("Stream maximum age must be positive")
        self.max_age=max_age
        self.lock=Lock()
        self.sample=None

    def put(self, captured_s, result):
        if not np.isfinite(captured_s):
            raise ValueError("Invalid monotonic capture timestamp")
        with self.lock:
            if self.sample and captured_s<=self.sample[0]:
                return False
            self.sample=(float(captured_s),result)
        return True

    def read(self, now):
        with self.lock:
            s=self.sample
        age=now-s[0] if s else np.inf
        if s is None or not 0<=age<=self.max_age:
            return None
        return s[1],s[0],age

    def reset(self):
        with self.lock:
            self.sample=None


class AppleMask:
    """A calibrated red apple tracker for the initial task, using RGB only."""
    def __init__(self, roi=None, min_pixels=35, max_jump_fraction=.17, stale_s=.25):
        self.roi=roi
        self.min_pixels=min_pixels
        self.max_jump=max_jump_fraction
        self.stale=stale_s
        self.last=None
        self.entered=None

    def __call__(self,rgb,t):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or not np.isfinite(t):
            raise ValueError("Expected timestamped camera RGB uint8")
        if self.entered is not None and t<=self.entered:
            raise ValueError("Out-of-order vision frame")
        self.entered=t
        h,w=rgb.shape[:2]
        s=np.asarray(rgb,dtype=np.int16)
        r,g,b=np.moveaxis(s,-1,0)
        mask=(r>52)&(r-g>24)&(r-b>18)&(r>1.21*g)&(r>1.16*b)
        if self.roi:
            x0,y0,x1,y1=self.roi
            if not (0<=x0<x1<=w and 0<=y0<y1<=h):
                raise ValueError("Camera ROI lies outside the calibrated image")
            area=np.zeros((h,w),bool)
            area[y0:y1,x0:x1]=True
            mask &= area
        structure=np.ones((3,3),bool)
        mask=ndimage.binary_opening(mask,structure)
        mask=ndimage.binary_closing(mask,structure)
        components,k=ndimage.label(mask)
        counts=np.bincount(components.ravel(),minlength=k+1)
        candidates=np.flatnonzero((counts>=self.min_pixels)&(counts<.07*h*w))
        candidates=candidates[candidates!=0]
        previous=self.last if self.last and t-self.last[0]<=self.stale else None
        choices=[]
        for j,(ys,xs) in enumerate(ndimage.find_objects(components),1):
            if j not in candidates:
                continue
            y,x=np.where(components[ys,xs]==j)
            point=np.array([xs.start+x.mean(),ys.start+y.mean()])
            dist=np.linalg.norm((point-previous[1])/[w,h]) if previous else 0.
            if dist<=self.max_jump:
                choices.append((counts[j]/(1+25*dist),j,point))
        if not choices:
            return dict(mask=np.zeros((h,w),bool),centroid=None,confidence=0.,
                        captured_s=t,source="red_component_tracker")
        score,j,point=max(choices,key=lambda x:x[0])
        self.last=(t,point)
        return dict(mask=(components==j),centroid=point,confidence=float(min(.7,.35+score/(.025*h*w))),
                    captured_s=t,source="red_component_tracker")


class Sam3Mask:
    """Run local prompt segmentation in the perception worker and associate its instances causally."""
    def __init__(self, model_dir, concept, device, roi=None, threshold=.5):
        import torch
        from transformers import AutoModel, AutoProcessor

        root=Path(model_dir).resolve()
        if not (root/"config.json").is_file() or not concept.strip() or not 0<threshold<1:
            raise ValueError("Supply local SAM 3 weights, a task concept, and a probability threshold")
        self.device,self.torch=device,torch
        self.model=AutoModel.from_pretrained(root,local_files_only=True).to(device).eval()
        if self.model.config.model_type!="sam3":
            raise ValueError("Segmentation checkpoint must be SAM 3")
        self.model.requires_grad_(False)
        self.processor=AutoProcessor.from_pretrained(root,local_files_only=True)
        self.source=f"SAM3:{root}"
        self.concept,self.roi,self.threshold=concept,roi,threshold
        self.last,self.seen=None,None

    def __call__(self,rgb,t):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or not np.isfinite(t):
            raise ValueError("Expected timestamped RGB uint8")
        if self.seen is not None and t<=self.seen:
            raise ValueError("Out-of-order SAM 3 image")
        self.seen=t
        h,w=rgb.shape[:2]
        with self.torch.inference_mode():
            packet=self.processor(images=rgb,text=self.concept,return_tensors="pt").to(self.device)
            detected=self.processor.post_process_instance_segmentation(self.model(**packet),
                threshold=self.threshold,target_sizes=[(h,w)])[0]
        old=self.last if self.last and t-self.last[0]<=.25 else None
        choices=[]
        for mask,score in zip(detected["masks"].cpu().numpy(),detected["scores"].cpu().numpy()):
            mask=mask.astype(bool)
            y,x=np.where(mask)
            if len(x)<35:
                continue
            point=np.array([x.mean(),y.mean()])
            if self.roi and not (self.roi[0]<=point[0]<self.roi[2] and self.roi[1]<=point[1]<self.roi[3]):
                continue
            if old and np.linalg.norm((point-old[2])/[w,h])>.17:
                continue
            overlap=np.count_nonzero(mask&old[1])/np.count_nonzero(mask|old[1]) if old else 0.
            choices.append((float(score)+overlap,mask,point,float(score)))
        if not choices:
            return dict(mask=np.zeros((h,w),bool),centroid=None,confidence=0.,captured_s=t,source=self.source)
        _,mask,point,confidence=max(choices,key=lambda x:x[0])
        self.last=(t,mask,point)
        return dict(mask=mask,centroid=point,confidence=confidence,captured_s=t,source=self.source)


class DepthTrack:
    """Lift an associated mask into measured partial geometry and a center estimate."""
    def __init__(self,K,base_from_camera,diameter_m):
        if not np.isfinite(diameter_m) or diameter_m<=0:
            raise ValueError("Center fitting requires an independent metric diameter")
        self.K,self.transform,self.radius=np.asarray(K),np.asarray(base_from_camera),diameter_m/2
        self.previous=None

    def __call__(self,depth_m,track,t,pose_estimate=None):
        if abs(track["captured_s"]-t)>.012:
            raise ValueError("Mask and depth must refer to the same capture")
        pts=unproject(depth_m,track["mask"],self.K,self.transform)
        if len(pts)<32 or track["confidence"]<=0.:
            self.previous=None
            return None
        pts=pts[np.linspace(0,len(pts)-1,min(len(pts),512),dtype=int)]
        if pose_estimate:
            pose=np.asarray(pose_estimate["pose"],dtype=float)
            if (pose.shape!=(7,) or np.any(~np.isfinite(pose)) or abs(np.linalg.norm(pose[3:])-1)>1e-5
                    or not np.isfinite(pose_estimate["captured_s"]) or abs(pose_estimate["captured_s"]-t)>.012
                    or not pose_estimate["source"]):
                raise ValueError("Expected a calibrated base-frame xyz/wxyz estimate")
            rot_ok=True
            conf=min(track["confidence"],pose_estimate["confidence"])
            source=pose_estimate["source"]
            fit_error=None
        else:
            center=np.median(pts,axis=0)
            axis=center-self.transform[:3,3]
            initial=center+axis/np.linalg.norm(axis)*self.radius
            residual=lambda x: np.linalg.norm(x-pts,axis=1)-self.radius
            fit=least_squares(residual,initial,loss="soft_l1",f_scale=.003,max_nfev=25)
            fit_error=float(np.median(abs(fit.fun)))
            singular=np.linalg.svd(fit.jac,compute_uv=False)
            if (not fit.success or fit_error>.007 or singular[-1]<.08*singular[0]
                    or np.linalg.norm(fit.x-initial)>self.radius):
                self.previous=None
                return None
            pose=np.r_[fit.x,[1.,0.,0.,0.]]
            rot_ok=False  # An apple center and a sphere fit do not identify orientation.
            conf=min(.55,track["confidence"])*(1-fit_error/.01)
            source="robust fixed-radius fit to measured depth"
        if not 0.<conf<=1.:
            raise ValueError("Invalid position estimate confidence")
        coast=self.previous
        if coast and t<=coast["time"]:
            raise ValueError("Nonincreasing depth clock")
        dt=t-coast["time"] if coast else np.inf
        linear_ok=dt<=.15
        angular_ok=bool(linear_ok and rot_ok and coast["orientation_valid"])
        v=(pose[:3]-coast["pose"][:3])/dt if linear_ok else np.zeros(3)
        ang=(rotation(pose[3:])*rotation(coast["pose"][3:]).inv()).as_rotvec()/dt if angular_ok else np.zeros(3)
        value=dict(pose=pose,twist=np.r_[v,ang],orientation_valid=rot_ok,linear_valid=linear_ok,
                   angular_valid=angular_ok,confidence=conf,time=t,source=source,fit_error_m=fit_error,
                   geometry=partial_points(pts,pose,conf))
        self.previous=value
        return value


@dataclass
class WireOrder:
    """Declare the measured name, sign and zero calibration of each SDK actuator."""
    permutation: np.ndarray
    signs: np.ndarray
    zero_rad: np.ndarray
    lo_rad: np.ndarray
    hi_rad: np.ndarray
    speed_rad_s: np.ndarray

    @classmethod
    def from_calibration(cls,names,meta):
        names=list(names)
        order=meta["sdk_joint_names"]
        if len(names)!=43 or len(order)!=43 or len(set(order))!=43 or set(names)!=set(order):
            raise ValueError("All 29 Body and both 7-joint hand channels require a named calibration")
        if any(set(order[a:b])!=set(names[a:b]) for a,b in ((0,29),(29,36),(36,43))):
            raise ValueError("SDK body, left, and right channels must retain their actuator banks")
        fields=[np.asarray(meta[k],dtype=float) for k in ("signs","zero_rad","lo_rad","hi_rad","speed_rad_s")]
        s,_,lo,hi,sp=fields
        if (any(x.shape!=(43,) or not np.isfinite(x).all() for x in fields)
                or not np.isin(s,[-1.,1.]).all() or np.any(lo>=hi) or np.any(sp<=0)):
            raise ValueError("Invalid SDK radians, limits or speeds")
        return cls(np.array([names.index(n) for n in order]),*fields)

    def state(self,q_wire,qd_wire=None):
        if np.shape(q_wire)!=(43,) or not np.isfinite(q_wire).all() or (
            qd_wire is not None and (np.shape(qd_wire)!=(43,) or not np.isfinite(qd_wire).all())):
            raise ValueError("Invalid SDK joint position or velocity packet")
        inverse=np.argsort(self.permutation)
        return ((self.signs*(q_wire-self.zero_rad))[inverse],
                None if qd_wire is None else (self.signs*qd_wire)[inverse])

    def command(self,q_canonical,previous_wire,dt):
        if q_canonical.shape!=(43,) or previous_wire.shape!=(43,) or dt!=.02:
            raise ValueError("Expected a named 43-axis 50 Hz target and previous applied command")
        if not np.isfinite(q_canonical).all() or not np.isfinite(previous_wire).all():
            raise ValueError("Nonfinite hardware target")
        if np.any(previous_wire<self.lo_rad) or np.any(previous_wire>self.hi_rad):
            raise ValueError("Previous hardware command outside calibrated range")
        q=self.signs*q_canonical[self.permutation]+self.zero_rad
        q=np.clip(q,previous_wire-self.speed_rad_s*dt,previous_wire+self.speed_rad_s*dt)
        wire=np.clip(q,self.lo_rad,self.hi_rad)
        return dict(sdk_body29=wire[:29],sdk_left7=wire[29:36],sdk_right7=wire[36:],
                    sdk_target43=wire,issued_canonical43=self.state(wire)[0])


def join_joint_packets(streams, mapping, now, max_skew_s):
    """Join stamped Body/left/right SDK encoder results without interpolation."""
    selected=[s.read(now) for s in streams]
    if len(selected)!=3 or any(s is None for s in selected):
        return None
    samples,times,_=zip(*selected)
    if max(times)-min(times)>max_skew_s:
        return None
    for x,n in zip(samples,(29,7,7)):
        if np.shape(x["q"])!=(n,) or ("qd" in x and np.shape(x["qd"])!=(n,)):
            raise ValueError("SDK joint packet banks must be Body29, left7, right7")
    q=np.concatenate([x["q"] for x in samples])
    qd=np.concatenate([x["qd"] for x in samples]) if all("qd" in x for x in samples) else None
    q,qd=mapping.state(q,qd)
    return dict(q=q,qd=qd,capture_by_bank_s=np.array(times)),min(times),now-min(times)


class FixedBaseFK:
    """Kinematics only; no simulator physics or object truth in these inputs."""
    def __init__(self,m,names):
        self.m=m
        self.d=mj.MjData(m)
        self.joint_q=np.array([int(m.joint(n).qposadr[0]) for n in names])
        self.joint_v=np.array([int(m.joint(n).dofadr[0]) for n in names])
        self.base=m.body("pelvis").id
        self.wrist=m.body("left_wrist_yaw_link").id
        self.tips=[m.body(f"left_hand_{g}_{2 if g=='thumb' else 1}_link").id for g in GROUPS]
        mj.mj_kinematics(m,self.d)
        self.origin=self.d.xpos[self.base].copy()
        self.axes=rotation(self.d.xquat[self.base]).inv()

    def measure(self,q,qd=None):
        d=self.d
        d.qpos[self.joint_q]=q
        d.qvel[self.joint_v]=0. if qd is None else qd
        mj.mj_kinematics(self.m,d)
        mj.mj_comPos(self.m,d)
        mj.mj_comVel(self.m,d)
        v=np.empty(6)
        mj.mj_objectVelocity(self.m,d,mj.mjtObj.mjOBJ_XBODY,self.wrist,v,0)
        p=self.axes.apply(d.xpos[self.wrist]-self.origin)
        return np.r_[p,wxyz(self.axes*rotation(d.xquat[self.wrist]))],np.r_[self.axes.apply(v[3:]),self.axes.apply(v[:3])],self.axes.apply(
            d.xpos[self.tips]-self.origin)


def nominal_from_capture(policy, fk, joint, camera, root, tick, now, task, seed, previous=None):
    """Planner-worker boundary. Joins fresh encoder/RGB captures and preserves raw Body40/Hand14."""
    if joint is None or camera is None or abs(joint[1]-camera[1])>.035:
        raise ValueError("No synchronized encoder and RGB capture for a nominal plan")
    if not (0<=now-joint[1]<=.06 and 0<=now-camera[1]<=.12):
        raise ValueError("Nominal planning inputs are stale or in the future")
    q=np.asarray(joint[0]["q"])
    root=np.asarray(root)
    if root.shape!=(7,) or not np.isfinite(root).all() or abs(np.linalg.norm(root[3:])-1)>1e-5:
        raise ValueError("Movens fixed-base root must be calibrated xyz/wxyz in the planner frame")
    plan=policy.predict(camera[0],np.r_[q[:29],root],q[29:],tick,previous_plan=previous,seed=seed,task_text=task)
    targets=np.c_[plan["body_absolute"][:,:29],plan["hand_absolute"]]
    wrist=np.array([fk.measure(target)[0] for target in targets])
    return plan,targets,wrist


def observation(joint, pose, fk, priors, now, *, target=None, contacts=None):
    """Build the common sensor contract from results selected by capture age."""
    if joint is None:
        return None,empty_cloud()
    sample,ts,age=joint
    q,qd=sample["q"],sample.get("qd")
    wrist,wrist_twist,tips=fk.measure(q,qd)
    dims=priors["dimensions"]
    p, pose_ts, p_age = pose if pose is not None else (None,None,0.)
    q_ok=qd is not None
    age=max(age,p_age)
    cf=p["confidence"]*max(.05,1-p_age/.15) if p else 0.
    c=contacts or {}
    support="original_support" in c and "target_support" in c
    geometry=p["geometry"] if p else empty_cloud()
    tip_values=np.zeros((3,4))
    if p and np.any(geometry["point_valid"]):
        points=geometry["points"][geometry["point_valid"],:3]
        cloud=rotation(p["pose"][3:]).apply(points)+p["pose"][:3]
        delta=cloud[None]-tips[:,None]
        norm=np.linalg.norm(delta,axis=-1)
        chosen=np.argmin(norm,axis=1)
        distances=norm[np.arange(3),chosen]
        tip_values=np.c_[distances,delta[np.arange(3),chosen]/np.maximum(distances[:,None],1e-8)]
    target_pose=target["pose"] if target else np.array([0.,0.,0.,1.,0.,0.,0.])
    margin=target.get("object_margin_m") if target else None
    return dict(q=q,qd=qd if q_ok else np.zeros(43),qd_valid=q_ok,q_valid=True,
                q_confidence=max(.05,1-(now-ts)/.08),q_time_s=ts,
                wrist=wrist,wrist_twist=wrist_twist,wrist_valid=True,wrist_twist_valid=q_ok,
                object=p["pose"] if p else np.array([0.,0.,0.,1.,0.,0.,0.]),object_valid=p is not None,
                object_confidence=cf,object_time_s=pose_ts,orientation_valid=bool(p and p["orientation_valid"]),
                linear_valid=bool(p and p["linear_valid"]),angular_valid=bool(p and p["angular_valid"]),
                twist=p["twist"] if p else np.zeros(6),dims=dims["value"] if dims else np.zeros(3),
                dims_valid=dims is not None,dims_confidence=dims["confidence"] if dims else 0.,
                diameter=max(dims["value"]) if dims else 0.,diameter_valid=dims is not None,
                target=target_pose,target_valid=target is not None,
                target_confidence=target["confidence"] if target else 0.,
                margin=margin if margin is not None else 0.,margin_valid=margin is not None,
                tips=tip_values,tips_valid=bool(geometry["point_valid"].any()),tips_confidence=cf,
                groups=c.get("groups",np.zeros(3,bool)),groups_valid=c.get("groups_valid",np.zeros(3,bool)),
                support_valid=support,original_support=c.get("original_support",False),
                target_support=c.get("target_support",False),robot_contact_valid="robot_contact" in c,
                robot_contact=c.get("robot_contact",False),time=now,age_s=age,confidence=cf),geometry


def hardware_phase(phase, now, obs, closing, *, stable_interval=None):
    # Establishing a grasp needs reliable relative velocity, object size and
    # direct/proven contact evidence. A stale tracker cannot qualify a hold.
    cues=dict(obs)
    cues["wrist_valid"]=(obs["wrist_valid"] and obs["wrist_twist_valid"]
                          and obs["linear_valid"] and obs["diameter_valid"])
    return phase.advance(now,cues,closing,stable_interval=stable_interval)


def hardware_features(obs, targets, planned_wrist, previous_action, previous_correction, phase,
                      task_left, chunk_age, *, ablation="phase"):
    """Reuse learned layouts; separately mask orientation and measured rates."""
    a=actor_frame(obs,targets,planned_wrist,previous_action,previous_correction,phase,task_left,chunk_age,
                  ablation=ablation)
    d=dyn_frame(obs,previous_action)
    for features,is_actor in ((a,True),(d,False)):
        twist_valid=not is_actor or obs["wrist_twist_valid"]
        for sl,valid in ((slice(89,95),obs["orientation_valid"]),
                         (slice(95,98),obs["linear_valid"] and twist_valid),
                         (slice(98,101),obs["angular_valid"] and twist_valid)):
            if not valid:
                features[:,sl]=0.
    return a,d

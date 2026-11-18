"""Use prepared nominal blocks and calibrated sensor results at the control boundary."""
from collections import deque

import numpy as np
import torch

from .contract import ActionLimits, LEFT
from .data import closure
from .features import window
from .hardware import hardware_features, hardware_phase
from .models import Q, gather, load_models
from .phase import Phase
from .pretrain import ensemble


class RobotPolicy:
    """Fixed base, left hand, D0. No transport, perception or planner I/O in a decision."""

    def __init__(self, cfg, checkpoint, joint_names, wire, device, *, offline=False):
        self.actor,self.dynamics,small=load_models(checkpoint,device)
        if (list(small["joint_names"])!=list(joint_names) or len(small.get("critics",[]))!=2
                or not offline and not small.get("identity",{}).get("deployment")):
            raise ValueError("Expected compatible actor, both SAC critics and deployment-trained weights")
        self.critics=[Q().to(device).eval() for _ in small["critics"]]
        for net,weights in zip(self.critics,small["critics"]):
            net.load_state_dict(weights)
        self.wire,self.device,self.offline=wire,device,offline
        first,last=wire.state(wire.lo_rad)[0],wire.state(wire.hi_rad)[0]
        lo,hi=np.minimum(first,last),np.maximum(first,last)
        scale=(hi[LEFT]-lo[LEFT])*cfg["residual_scale_fraction"]
        self.limits=ActionLimits(lo,hi,scale,.5*scale)
        self.dt=cfg["control_dt"]
        if self.dt!=.02:
            raise ValueError("The robot position interface requires a 50 Hz control schedule")
        self.rules,self.phase=cfg["phase"],Phase(cfg["phase"])
        self.history=small.get("identity",{}).get("history") or cfg["history"]
        self.a_frames,self.d_frames,self.times=(deque(maxlen=self.history) for _ in range(3))
        self.tick,self.block,self.last,self.start=-1,None,None,None

    @torch.inference_mode()
    def decide(self, now, tick, obs, cloud, physical, previous_wire, *, task_left_s,
               block=None, ready_s=None, active=True):
        """Return a bounded proposal. In a live caller previous_wire is the applied target."""
        start=now if self.start is None else self.start
        if (tick!=self.tick+1 or abs(now-start-tick*self.dt)>.002 or not np.isfinite(now)
                or self.times and now<=self.times[-1]):
            raise ValueError("Missed or invalid control boundary; start a new recorded attempt")
        if (obs is None or not obs["q_valid"] or not 0<=now-obs["q_time_s"]<=.06
                or obs["object_valid"] and not 0<=now-obs["object_time_s"]<=.15):
            raise ValueError("No fresh encoder/causal perception result for this control boundary")
        if active and self.last is not None and not self.last["active"]:
            raise ValueError("Task decisions cannot restart in the release hold tail")
        if tick%20==0 and active:
            if block is None or ready_s is None or not 0<=now-ready_s<=.4:
                raise ValueError("Planner block was not available by its execution boundary")
            raw,targets,wrist=block
            if (raw["first_physical_frame"]!=tick+1 or raw["prefix_length"]!=(10 if tick else 0)
                    or raw["plan_id"]!=tick//20 or np.shape(targets)!=(30,43) or np.shape(wrist)!=(30,7)):
                raise ValueError("Wrong nominal block, canonical joint order or committed prefix")
            self.block=block
        elif self.block is None or block is not None:
            raise ValueError("A prepared nominal block may enter only at its fixed boundary")
        raw,targets,wrist=self.block
        offset=tick%20 if active else self.last["plan_offset"]
        nominal=targets[offset].astype(float)
        applied=self.wire.state(previous_wire)[0]
        if self.last and not self.offline and not np.allclose(previous_wire,self.last["sdk_target43"],rtol=0,atol=1e-6):
            raise ValueError("The independently applied target differs from the previous issued position target")
        previous_nominal=self.last["nominal"] if self.last else applied
        correction=None if self.last is None or self.offline else applied[LEFT]-self.last["limited_nominal"][LEFT]
        close=bool(closure(nominal,previous_nominal,self.dt,self.rules))
        phase=hardware_phase(self.phase,now,obs,close)
        af,df=hardware_features(obs,targets[offset:],wrist[offset:],None if self.offline else applied,
                                correction,phase,task_left_s,offset*self.dt)
        self.a_frames.append(af)
        self.d_frames.append(df)
        self.times.append(now)
        times=np.asarray(self.times)
        def inputs(frames):
            h,n=window(np.asarray(frames),times,len(times)-1,self.history)
            return dict(history=h,length=n,physical=physical,**cloud)
        actor_context,dynamics_context=inputs(self.a_frames),inputs(self.d_frames)
        z=self.actor.sample(gather([actor_context],self.device),True)[0][0].cpu().numpy().astype(float) if active else np.zeros(7)
        zero_corr=np.zeros(7)
        desired,_=self.limits.execute(nominal,z,zero_corr if correction is None else correction)
        base=self.wire.command(np.clip(nominal,self.limits.lo,self.limits.hi),previous_wire,self.dt)
        command=self.wire.command(desired if active else applied,previous_wire,self.dt)
        self.start,self.tick=start,tick
        self.last=dict(tick=tick,time_s=now,nominal=nominal,plan_id=int(raw["plan_id"]),plan_offset=offset,active=active,
            limited_nominal=base["issued_canonical43"],actor_proposal=z,**command)
        return dict(self.last,phase=phase,actor_frame=af,dynamics_frame=df,
                    actor_context=actor_context,dynamics_context=dynamics_context)

    @torch.inference_mode()
    def diagnostics(self, decision, target_wire, dt):
        """D0 is downstream of the command. Live callers pass independently applied targets."""
        if not .018<=dt<=.022:
            raise ValueError("Expected measured duration for a 50 Hz command interval")
        target=self.wire.state(target_wire)[0]
        z=torch.as_tensor(decision["actor_proposal"].reshape(1,7),dtype=torch.float32,device=self.device)
        b=gather([decision["actor_context"]],self.device)
        q=torch.stack([m(b,z) for m in self.critics])[:,0]
        b=gather([dict(decision["dynamics_context"],action=target,dt=[dt])],self.device)
        mean,var,spread,event=ensemble(self.dynamics,b)
        return {k:x.cpu().numpy() for k,x in zip(("q_values","d0_delta","d0_variance","d0_disagreement","d0_events"),
                (q,mean[0],var[0],spread[0],event[0]))}

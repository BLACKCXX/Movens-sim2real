"""Separate actual MuJoCo randomization from a render/measurement adapter."""
from collections import deque
from copy import deepcopy

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import rotation, wxyz
from .features import packed, point_context
from .hardware import DepthTrack, FixedBaseFK, observation
from .vision import empty_cloud


PROTOCOL="s1-candidate-observation-v3"


def factory(mode, index=0):
    if mode=="curriculum":
        mode=("physics","noisy","deployment","deployment")[index%4]
    if mode not in ("physics","noisy","deployment"):
        raise ValueError("Unknown S1 observation profile")
    return lambda scene, seed: SimObservation(scene,seed,mode)


class SimObservation:
    def __init__(self,scene,seed,mode):
        self.scene=scene
        self.s=scene.sensors
        self.rng=np.random.default_rng(seed)
        self.mode=mode
        self.delay=0 if mode=="physics" else int(seed%2)
        self.samples=deque(maxlen=3)
        self.image_pose=None
        self.geometry=empty_cloud()
        self.drop_until=-1.
        self.dropped=0
        self.plate_shift=np.zeros(3)
        self.apple_shift=np.zeros(3)
        self.mass_scale=self.friction_hand=self.friction_plate=1.
        self.yaw=0.
        self.calibration_shift=np.zeros(3)
        self.camera_error=np.zeros(3)
        self.exposure=1.
        if mode in ("physics","deployment"):
            self.plate_shift[:2]=self.rng.uniform(-.012,.012,2)
            self.apple_shift[:2]=self.rng.uniform(-.008,.008,2)
            self.yaw=self.rng.uniform(-.08,.08)
            self.mass_scale,self.friction_hand=self.rng.uniform(.9,1.1,2)
            self.friction_plate=self.friction_hand
            self.randomize()
        if mode!="physics":
            self.calibration_shift=self.rng.normal(0,.002,3)
            self.camera_error=self.rng.normal(0,.01,3)
            self.exposure=float(self.rng.uniform(.85,1.15))
        if mode=="deployment":
            self.fk=FixedBaseFK(scene.m,self.s.names)
            self.dimensions=self.s.dims*self.rng.uniform(.97,1.03,3)
            self.dimension_prior=dict(dimensions=dict(value=self.dimensions,confidence=.7))
            self.velocity_gap_offset=int(seed%40)
            self.target_error=self.rng.normal(0,.002,3)
        self.identity=dict(protocol=PROTOCOL,mode=mode,seed=seed,observation_delay_ticks=self.delay,
                           command_delay_ticks=0,
                           mass_scale=self.mass_scale,hand_object_friction_scale=self.friction_hand,
                           plate_object_friction_scale=self.friction_plate,plate_shift_m=self.plate_shift,
                           plate_yaw_rad=self.yaw,apple_shift_m=self.apple_shift,
                           camera_calibration_offset_m=self.calibration_shift,camera_axis_angle_rad=self.camera_error,
                           exposure=self.exposure,hidden_physical=mode=="deployment",
                           groups="geometric and motion proxy; no true contact inputs" if mode=="deployment" else "native",
                           geometry="rendered depth plus simulator-ID mask; partial" if mode=="deployment" else "complete",
                           perception=("depth center, no rotational evidence, finite-difference velocity, 240ms encoder velocity gaps"
                                       if mode=="deployment" else "simulator truth or numerically perturbed pose"),
                           estimated_dimensions_m=self.dimensions if mode=="deployment" else None,
                           target_measurement_offset_m=self.target_error if mode=="deployment" else None)

    def randomize(self):
        sc,s=self.scene,self.s
        m=sc.m
        for i in s.apple_bodies:
            m.body_mass[i]*=self.mass_scale
            m.body_inertia[i]*=self.mass_scale
        # Scale both endpoints of these contact classes: otherwise a lower
        # coefficient can be cancelled by MuJoCo's equal-priority max mix.
        a=set(s.apple_geoms)
        relevant=np.isin(m.geom_bodyid,[*s.robot_bodies,*s.plate_bodies,*s.apple_bodies])
        m.geom_friction[relevant]*=self.friction_hand
        for i,j in enumerate(m.pair_geom1):
            g=int(j)
            h=int(m.pair_geom2[i])
            other=h if g in a else g if h in a else None
            if other is not None:
                body=int(m.geom_bodyid[other])
                if body in s.robot_bodies:
                    m.pair_friction[i]*=self.friction_hand
                elif body in s.plate_bodies:
                    m.pair_friction[i]*=self.friction_plate
        m.body_pos[s.plate]+=self.plate_shift
        m.body_quat[s.plate]=wxyz(Rotation.from_rotvec([0,0,self.yaw])*rotation(m.body_quat[s.plate]))
        mj.mj_setConst(m,sc.d)
        mj.mj_kinematics(m,s.fk)
        s.target_world=np.r_[s.fk.xpos[s.plate],s.fk.xquat[s.plate]]
        s.geo["plate_pos"]=s.target_world[:3].copy()
        s.geo["plate_rotation"]=rotation(s.target_world[3:]).as_matrix()
        s.geo["apple_weight_N"]=float(m.body_subtreemass[s.apple]*-m.opt.gravity[2])
        s.physical,s.physical_valid=s.physics_parameters()

    def after_reset(self):
        if np.any(self.apple_shift):
            q=self.scene.d.qpos
            i=self.s.apple_q
            q[i:i+3]+=self.apple_shift
            self.scene.sim.synchronize()
        return self.scene.snapshot()

    def prepare_view(self,view):
        self.view=view
        m=view.model
        self.cam=m.camera("approx_ego_camera").id
        if self.mode=="deployment":
            # A collider can live under an unnamed descendant. The renderer
            # has its own compiled body IDs, so follow its Apple subtree.
            apple=np.zeros(m.nbody,bool)
            apple[m.body("apple_root").id]=True
            for i in range(1,m.nbody):
                apple[i]|=apple[m.body_parentid[i]]
            self.apple_view_geoms=np.flatnonzero(apple[m.geom_bodyid])
            if not len(self.apple_view_geoms):
                raise ValueError("No rendered apple geometry under apple_root")
        p=m.body("plate_root").id
        m.body_pos[p]=self.scene.m.body_pos[self.s.plate]
        m.body_quat[p]=self.scene.m.body_quat[self.s.plate]
        m.cam_pos[self.cam]+=self.calibration_shift
        m.cam_quat[self.cam]=wxyz(rotation(m.cam_quat[self.cam])*Rotation.from_rotvec(self.camera_error))
        m.light_diffuse[:]*=self.exposure

    def observe(self,t,true,rgb):
        self.samples.append((t,true))
        ts,last=self.samples[max(0,len(self.samples)-1-self.delay)]
        if self.mode=="physics":
            return dict(true,age_s=t-ts,time=t)
        if self.mode=="deployment":
            return self.observe_deployment(t,ts,last,rgb)
        s=self.s
        pw=last["object"].copy()
        pw[:3]+=self.rng.normal(0,.003,3)
        pw[3:]=wxyz(rotation(pw[3:])*Rotation.from_rotvec(self.rng.normal(0,.025,3)))
        # Perturb one pose and transform every derived quantity consistently.
        vw=last["twist"].copy()
        vw[:3]+=self.rng.normal(0,.004,3)
        raw=np.r_[s.base.apply(pw[:3])+s.origin,wxyz(s.base*rotation(pw[3:]))]
        world_vel=np.r_[s.base.apply(vw[:3]),rotation(pw[3:]).inv().apply(vw[3:])]
        contact={k:last[k] for k in ("groups","groups_valid","original_support","target_support","robot_contact",
                                      "support_forces","contact_counts")}
        res=s.observation(last["q"],last["qd"],raw,world_vel,contact)
        res.update(age_s=t-ts,time=t,object_confidence=.8)
        return res

    def observe_deployment(self,t,ts,last,rgb):
        if t>=self.drop_until and self.rng.random()<.018:
            self.drop_until=t+self.rng.uniform(.08,.22)
            self.image_pose=None
        if rgb is not None and t>=self.drop_until:
            self.new_depth(t,rgb.shape[:2])
        p=self.image_pose
        sensor=p is not None and t-p["time"]<=.12 and t>=self.drop_until
        self.dropped+=not sensor
        pose=(p,p["time"],t-p["time"]) if sensor else None
        q=last["q"]+self.rng.normal(0,.0007,43)
        velocity_valid=(round(t/.02)+self.velocity_gap_offset)%40>=12
        qd=last["qd"]+self.rng.normal(0,.003,43) if velocity_valid else None
        target=self.s.to_base(self.s.target_world)
        target[:3]+=self.target_error
        margin=None
        if sensor:
            center=rotation(target[3:]).inv().apply(p["pose"][:3]-target[:3])
            # A circumscribed radius gives a conservative footprint proxy when orientation is missing.
            margin=float(np.min(center[:2]@self.s.inner.T+self.s.hull_offsets)-max(self.dimensions)/2)
        ob,self.geometry=observation((dict(q=q,qd=qd),ts,t-ts),pose,self.fk,self.dimension_prior,t,
                                     target=dict(pose=target,confidence=.8,object_margin_m=margin))
        proxy=bool(sensor and p["linear_valid"] and velocity_valid and ob["tips_valid"])
        if proxy:
            speed=np.linalg.norm(ob["twist"][:3])
            rel=np.linalg.norm(ob["twist"][:3]-ob["wrist_twist"][:3])
            group=(ob["tips"][:,0]<.009) & (rel<.055)
            bottom=p["pose"][2]-self.dimensions[2]/2
            on_target=margin>=.001 and abs(target[2]-bottom)<.015 and speed<.02
            on_original=abs(bottom-(self.s.table_top-self.s.origin[2]))<.012 and not on_target and speed<.02
            ob.update(groups=group,groups_valid=np.ones(3,bool),robot_contact=bool(group.any()),
                      robot_contact_valid=True,support_valid=True,target_support=bool(on_target),
                      original_support=bool(on_original))
        return ob

    def new_depth(self,t,image_shape):
        v=self.view
        r=v.renderer
        # An actual visibility/occlusion path in simulation. Simulator
        # segmentation IDs are training evidence, not a real perception model.
        r.enable_segmentation_rendering()
        try:
            ids=v.image("ego")
        finally:
            r.disable_segmentation_rendering()
        r.enable_depth_rendering()
        try:
            depth=v.image("ego")
        finally:
            r.disable_depth_rendering()
        mask=np.isin(ids[...,0],self.apple_view_geoms)&(ids[...,1]==mj.mjtObj.mjOBJ_GEOM)
        h,w=image_shape
        focal=.5*h/np.tan(np.deg2rad(v.model.cam_fovy[self.cam])*.5)
        K=np.array([[focal,0,(w-1)/2],[0,focal,(h-1)/2],[0,0,1.]])
        tf=np.eye(4)
        base=self.s.base.inv()
        tf[:3,:3]=base.as_matrix()@v.data.cam_xmat[self.cam].reshape(3,3)@np.diag([1.,-1.,-1.])
        tf[:3,3]=base.apply(v.data.cam_xpos[self.cam]-self.s.origin-self.calibration_shift)
        tf[:3,:3]=tf[:3,:3]@Rotation.from_rotvec(-self.camera_error).as_matrix()
        tracker=DepthTrack(K,tf,max(self.dimensions))
        tracker.previous=self.image_pose
        self.image_pose=tracker(depth,dict(mask=mask,captured_s=t,confidence=.7),t)

    def context(self):
        if self.mode!="deployment":
            geometry=point_context(self.s.point_cloud)
            mask=self.s.physical_valid.copy()
            quality=np.ones(29)
        else:
            geometry=self.geometry
            mask=self.s.physical_valid.copy()
            mask[:16]=False
            mask[26:29]=False
            quality=np.full(29,.7)
        values=self.s.physical.copy()
        if self.mode=="deployment":
            values[19:22]=self.dimensions
        # Observation age lives with the measurements. The physical slot is
        # the actuator delay, which the unchanged native controller does not add.
        return geometry,packed(values,mask,quality)

    def state(self):
        return dict(rng=self.rng.bit_generator.state,samples=list(self.samples),image_pose=self.image_pose,
                    geometry=self.geometry,drop_until=self.drop_until,dropped=self.dropped)

    def load_state(self,x):
        self.rng.bit_generator.state=x["rng"]
        self.samples.extend(x["samples"])
        for k in ("image_pose","geometry","drop_until","dropped"):
            setattr(self,k,x[k])

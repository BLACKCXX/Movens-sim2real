"""Read-only reuse of the original fixed-base MuJoCo position controller."""

import copy
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull

from simulation.apple_batch_evaluate import geometry, placed_geometry
from simulation.common import AppleSimulation

from .contract import ActionLimits, GROUPS, read, rotation, sha256, wxyz


INTEGRATION = mj.mjtState.mjSTATE_INTEGRATION


def compact_model(xml):
    """Original kinematics/colliders for offline observations, never stepped."""
    root = ET.parse(xml).getroot()
    world = root.find("worldbody")
    for node in list(world):
        if node.get("name", "").startswith("pad_node_"):
            world.remove(node)
    root.remove(root.find("deformable"))
    # The source XML resolves all files through absolute paths.
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def up_to_speed(v, evaluation):
    return ((np.linalg.norm(v[..., :3], axis=-1) <= evaluation["final_max_speed_m_s"]) &
            (np.linalg.norm(v[..., 3:], axis=-1) <= evaluation["final_max_angular_speed_rad_s"]))


def stable_intervals(tt, phys_t, q, v, counts, plate_force, table_force, geo, ev, dt):
    """Every actual native step must satisfy the release witness."""
    if not np.allclose(np.diff(phys_t), dt, rtol=0, atol=1e-8):
        raise ValueError("Release checks require a continuous physical time axis")
    bins = np.rint((tt-phys_t[0])/dt).astype(int)
    if (bins < 0).any() or (bins >= len(phys_t)).any() or not np.allclose(phys_t[bins], tt, rtol=0, atol=1e-8):
        raise ValueError("Release interval boundaries not present in native log")
    weight = geo["apple_weight_N"]
    ratio = plate_force/weight
    point_ok = ((counts[:, 2] > 0) & (counts[:, 3] == 0) & (counts[:, 0] == 0) & (counts[:, 1] == 0)
                & (ratio >= ev["plate_min_support_ratio"]) & (table_force/weight <= ev["plate_max_other_support_ratio"])
                & up_to_speed(v, ev))
    run = np.r_[0, np.cumsum(~point_ok)]
    good = np.flatnonzero(run[bins[1:]+1]-run[bins[:-1]+1] == 0)
    result = np.zeros(len(tt), dtype=bool)
    if not len(good):
        return result
    steps = np.diff(bins)[good]
    ix = np.concatenate([np.arange(bins[i]+1, bins[i+1]+1) for i in good])
    placed = placed_geometry(q[ix], geo, ev)[:, 0].astype(bool)
    result[good+1] = np.add.reduceat(~placed, np.r_[0, np.cumsum(steps)[:-1]]) == 0
    return result


class Sensors:
    """Fixed base frame. MuJoCo free qvel is world linear, local angular."""

    def __init__(self, model, joint_names, candidate, evaluation):
        self.m = m = model
        self.evaluation = evaluation
        self.names = list(joint_names)
        joints = np.array([m.joint(n).id for n in self.names])
        self.qadr, self.vadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        self.pelvis = m.body("pelvis").id
        self.wrist = m.body("left_wrist_yaw_link").id
        self.fk = mj.MjData(m)
        mj.mj_kinematics(m, self.fk)
        self.origin = self.fk.xpos[self.pelvis].copy()
        self.base = rotation(self.fk.xquat[self.pelvis])
        for j in joints:
            if m.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
                raise ValueError("Robot action interface must consist of 43 hinges")
        self.geo = geometry(m)
        self.apple = m.body("apple_root").id
        self.plate = m.body("plate_root").id
        self.table = m.body("table_root").id
        self.apple_q = int(m.jnt_qposadr[m.joint("apple_freejoint").id])
        self.apple_v = int(m.jnt_dofadr[m.joint("apple_freejoint").id])
        self.point_cloud = self._surface_points()
        self.dims = np.ptp(self.geo["apple_vertices"], axis=0)
        self.diameter = float(max(self.dims))
        geom_body = [m.body(b).name for b in m.geom_bodyid]
        collision = m.geom_contype | m.geom_conaffinity
        self.apple_bodies, self.plate_bodies, self.robot_bodies, self.table_bodies = (
            self.descendants(i) for i in (self.apple, self.plate, self.pelvis, self.table))
        self.apple_geoms = np.flatnonzero(np.isin(m.geom_bodyid, list(self.apple_bodies)) & (collision != 0))
        self.plate_geoms = np.flatnonzero(np.isin(m.geom_bodyid, list(self.plate_bodies)) & (collision != 0))
        self.tip_geoms = [[i for i, b in enumerate(geom_body) if b == f"left_hand_{g}_{2 if g == 'thumb' else 1}_link"
                           and collision[i]] for g in GROUPS]
        if not len(self.apple_geoms) or any(not g for g in self.tip_geoms):
            raise ValueError("Collision geometry is missing for the apple or distal links")
        self.contact_groups = {m.body(i).id: j for j, g in enumerate(GROUPS) for i in range(m.nbody)
                               if m.body(i).name.startswith(f"left_hand_{g}_")}
        self.table_top = float(read(candidate/"scene/batch_construction.json")["actual_uncompressed_soft_top_m"])
        self.target_world = np.r_[self.fk.xpos[self.plate], self.fk.xquat[self.plate]]
        hull = self.geo["plate_hull"]
        edges = np.roll(hull, -1, axis=0)-hull
        self.inner = np.column_stack((-edges[:, 1], edges[:, 0]))/np.linalg.norm(edges, axis=1, keepdims=True)
        self.hull_offsets = -np.sum(self.inner*hull, axis=1)
        self.physical, self.physical_valid = self.physics_parameters()

    def descendants(self, body):
        ids = {body}
        for k, parent in enumerate(self.m.body_parentid):
            if parent in ids:
                ids.add(k)
        return ids

    def _surface_points(self):
        # Deterministic area sampling from the exterior collision hull. The
        # AppleToPlate asset marks the single rigid fruit as graspable.
        hull = ConvexHull(self.geo["apple_vertices"])
        verts = self.geo["apple_vertices"][hull.simplices]
        area = np.linalg.norm(np.cross(verts[:, 1]-verts[:, 0], verts[:, 2]-verts[:, 0]), axis=1)
        rng = np.random.default_rng(11)
        tri = verts[rng.choice(len(verts), 256, p=area/area.sum())]
        u = np.sqrt(rng.uniform(size=(256, 1)))
        v = rng.uniform(size=(256, 1))
        xyz = (1-u)*tri[:, 0]+u*(1-v)*tri[:, 1]+u*v*tri[:, 2]
        return np.c_[xyz, np.ones(len(xyz))].astype(np.float32)

    def physics_parameters(self):
        m = self.m
        i = self.apple
        d = self.fk
        frame = rotation(d.xquat[i]).inv()
        parts = sorted(self.apple_bodies)
        masses = m.body_mass[parts]
        p = frame.apply(d.xipos[parts]-d.xpos[i])
        com = np.average(p, weights=masses, axis=0)
        inertia = np.zeros((3, 3))
        for j, mass, pos in zip(parts, masses, p):
            r = (frame * rotation(d.xquat[j]) * rotation(m.body_iquat[j])).as_matrix()
            delta = pos-com
            inertia += (r*m.body_inertia[j]) @ r.T + mass*(np.eye(3)*(delta@delta)-np.outer(delta, delta))
        # General contact mixing is determined by active pairs. Retain the
        # geometry-defined coefficient only if all relevant pairings agree.
        valid_friction = [True]*2
        coefficients = []
        for k, other in enumerate(([g for sub in self.tip_geoms for g in sub], self.plate_geoms)):
            pairs = []
            for a in self.apple_geoms:
                for b in other:
                    pa, pb = m.geom_priority[a], m.geom_priority[b]
                    pairs.append(m.geom_friction[a] if pa > pb else m.geom_friction[b] if pb > pa else
                                 np.maximum(m.geom_friction[a], m.geom_friction[b]))
            c = pairs[0]
            explicit = {(int(a), int(b)) for a, b in zip(m.pair_geom1, m.pair_geom2)}
            def pair_controls(a, b):
                return (int(a), int(b)) in explicit or (int(b), int(a)) in explicit
            valid_friction[k] = np.allclose(pairs, c) and not any(pair_controls(a, b) for a in self.apple_geoms for b in other)
            coefficients.append(c if valid_friction[k] else np.zeros(3))
        gravity = self.base.inv().apply(m.opt.gravity)
        normal = self.base.inv().apply([0., 0., 1.])
        offset = -normal@self.base.inv().apply(np.array([0., 0., self.table_top])-self.origin)
        params = np.r_[m.body_subtreemass[i], com, inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                       coefficients[0], coefficients[1], gravity, self.dims, normal, offset, 0., 1., 1.]
        mask = np.ones(29, dtype=bool)
        mask[10:13], mask[13:16] = valid_friction
        return params.astype(np.float32), mask

    def to_base(self, p):
        return np.r_[self.base.inv().apply(p[:3]-self.origin), wxyz(self.base.inv()*rotation(p[3:]))]

    def twist_base(self, p, vel):
        return np.r_[self.base.inv().apply(vel[:3]), (self.base.inv()*rotation(p[3:])).apply(vel[3:])]

    def pose_twist(self, data, body):
        v = np.empty(6)
        mj.mj_objectVelocity(self.m, data, mj.mjtObj.mjOBJ_XBODY, body, v, 0)
        return (self.to_base(np.r_[data.xpos[body], data.xquat[body]]),
                np.r_[self.base.inv().apply(v[3:]), self.base.inv().apply(v[:3])])

    def margin(self, p):
        geo = self.geo
        points = rotation(p[3:]).apply(geo["apple_vertices"])+p[:3]
        local = (points-geo["plate_pos"])@geo["plate_rotation"]
        # The minimum of the apple's entire projected collision footprint.
        return float(np.min(local[:, :2]@self.inner.T + self.hull_offsets))

    def fingertip_features(self, d):
        out = []
        for geoms in self.tip_geoms:
            best = 2.
            direction = np.zeros(3)
            for tip in geoms:
                for a in self.apple_geoms:
                    seg = np.empty(6)
                    distance = mj.mj_geomDistance(self.m, d, tip, a, 2., seg)
                    if distance < best:
                        best = distance
                        delta = seg[3:]-seg[:3]
                        norm = np.linalg.norm(delta)
                        direction = self.base.inv().apply(delta/norm) if norm > 1e-9 else np.zeros(3)
            out.append(np.r_[max(0., best), direction])
        return np.asarray(out)

    def observation(self, q, qd, pw, vw, contacts):
        # Scratch FK is independent of integration/constraint warm start.
        d = self.fk
        d.qpos[self.qadr], d.qvel[self.vadr] = q, qd
        d.qpos[self.apple_q:self.apple_q+7] = pw
        d.qvel[self.apple_v:self.apple_v+6] = vw
        mj.mj_kinematics(self.m, d)
        mj.mj_comPos(self.m, d)
        mj.mj_comVel(self.m, d)
        wrist, wrist_twist = self.pose_twist(d, self.wrist)
        return dict(q=q, qd=qd, object=self.to_base(pw), twist=self.twist_base(pw, vw),
                    wrist=wrist, wrist_twist=wrist_twist, tips=self.fingertip_features(d),
                    target=self.to_base(self.target_world), dims=self.dims, diameter=self.diameter,
                    margin=self.margin(pw), **contacts)

    def wrist_plan(self, plan29):
        d = self.fk
        poses = []
        for q in plan29:
            d.qpos[self.qadr[:29]] = q
            mj.mj_kinematics(self.m, d)
            poses.append(self.to_base(np.r_[d.xpos[self.wrist], d.xquat[self.wrist]]))
        return np.asarray(poses)

    def settle_support(self, d):
        """Solved forces from the last step, before refreshing kinematics."""
        groups = np.zeros(3, dtype=bool)
        counts = np.zeros(4, dtype=np.int32)  # pad, table, plate, robot
        up = np.zeros(2)  # table including pad, plate
        pair_force = np.zeros(6)
        m = self.m
        for i, c in enumerate(d.contact):
            if c.efc_address < 0 or c.exclude:
                continue
            g = c.geom
            a = -1 if g[0] < 0 else int(m.geom_bodyid[g[0]])
            b = -1 if g[1] < 0 else int(m.geom_bodyid[g[1]])
            if (a in self.apple_bodies) == (b in self.apple_bodies):
                continue
            which = 1 if a in self.apple_bodies else 0
            other = b if which else a
            mj.mj_contactForce(m, d, i, pair_force)
            local_up = pair_force[:3]@c.frame.reshape(3, 3)[:, 2]
            force_z = -local_up if which else local_up
            loaded = bool(pair_force[0] > self.evaluation["loaded_contact_force_N"])
            if other in self.plate_bodies:
                counts[2] += loaded
                up[1] += force_z
            elif c.flex[which] >= 0 or other in self.table_bodies:
                counts[0 if c.flex[which] >= 0 else 1] += loaded
                up[0] += force_z
            elif other in self.robot_bodies:
                counts[3] += loaded
                if other in self.contact_groups:
                    groups[self.contact_groups[other]] |= loaded
        return groups, counts, up

    def contact_obs(self, groups, counts, up, groups_valid):
        weight = self.geo["apple_weight_N"]
        support = counts[2] > 0 and up[1] >= self.evaluation["plate_min_support_ratio"]*weight
        return dict(groups=groups, groups_valid=groups_valid, original_support=bool(counts[0]+counts[1]),
                    target_support=bool(support), robot_contact=bool(counts[3]),
                    support_forces=up, contact_counts=counts)


class NativeScene:
    def __init__(self, candidate, cfg):
        self.path = candidate
        self.report = read(candidate/"run/report.json")
        self.evaluation = read(candidate/"evaluation.json")["evaluation_config"]
        reset = copy.deepcopy(read(candidate/"scene/robot.json")["reset_protocol"])
        reset["settle_steps"] = 0
        self.sim = AppleSimulation(candidate/"scene/scene.json", reset_protocol=reset)
        self.m, self.d = self.sim.model, self.sim.data
        self.dt = float(self.m.opt.timestep)
        self.interval = self.sim.substeps
        ids = self.sim.actuator_ids
        if (self.dt != cfg["physics_dt"] or not np.isclose(self.dt*self.interval, cfg["control_dt"], rtol=0, atol=1e-12)
                or not np.array_equal(self.sim.joint_ids, self.m.actuator_trnid[ids, 0])
                or not np.all(self.m.actuator_gear[ids, 0] == 1)
                or self.m.na or self.m.nu != 43):
            raise ValueError(f"{candidate}: native timestep/mapping/actuator protocol differs")
        self.sensors = Sensors(self.m, self.sim.joint_names, candidate, self.evaluation)
        self.limits = ActionLimits.from_scene(self, cfg["residual_scale_fraction"], cfg["control_dt"])
        self.end = float(self.report["original_trajectory_end_s"])
        self.tail = float(self.report["tail_verification_seconds"])
        self.settle = float(self.report["settle_duration_s"])

    def snapshot(self):
        a = np.empty(mj.mj_stateSize(self.m, INTEGRATION))
        mj.mj_getState(self.m, self.d, a, INTEGRATION)
        return a

    def restore(self, a, *, resume=False):
        if a.shape != (mj.mj_stateSize(self.m, INTEGRATION),) or not np.isfinite(a).all():
            raise ValueError("Incompatible integration snapshot")
        # New MjData clears stale warnings and derived solver diagnostics.
        self.d = self.sim.data = mj.MjData(self.m)
        mj.mj_setState(self.m, self.d, a, INTEGRATION)
        if resume:
            # Rebuild the soft-contact/solver workspace before continuation.
            # Forward can update warmstart values, so put back the saved
            # integration state after the workspace is initialized.
            mj.mj_forward(self.m, self.d)
            mj.mj_setState(self.m, self.d, a, INTEGRATION)
        else:
            self.sim.synchronize()
        if not np.array_equal(self.snapshot(), a):
            raise ValueError("Integration snapshot changed on restore")

    def restore_original(self, t):
        from .contract import exact_indices
        with np.load(self.path/"run/states.npz") as z:
            i = exact_indices(z["timestamps"], [t], self.path)[0]
            state = z["integration_state"][i]
        self.restore(state)
        return state

    def physical_state(self):
        s, d = self.sensors, self.d
        return (d.qpos[s.qadr].copy(), d.qvel[s.vadr].copy(),
                d.qpos[s.apple_q:s.apple_q+7].copy(), d.qvel[s.apple_v:s.apple_v+6].copy())

    def observe(self, evidence=None):
        s = self.sensors
        groups, counts, up = s.settle_support(self.d) if evidence is None else evidence
        return s.observation(*self.physical_state(), s.contact_obs(groups, counts, up, np.ones(3, dtype=bool)))

    def step(self, target, *, observe_stability=False, guard=None):
        d, m = self.d, self.m
        begin = float(d.time)
        if target.shape != (43,) or not np.isfinite(target).all() or np.any(target < self.limits.lo-1e-7) or np.any(target > self.limits.hi+1e-7):
            raise ValueError("Attempt to execute an invalid joint target")
        d.ctrl[self.sim.actuator_ids] = target
        stability, observations = False, []
        if observe_stability or guard is not None:
            for _ in range(self.interval):
                mj.mj_step(m, d)
                if guard is not None:
                    error = guard.check(float(d.time))
                    if error:
                        raise FloatingPointError(error)
                if observe_stability:
                    c = self.sensors.settle_support(d)
                    q = self.sensors.apple_q
                    v = self.sensors.apple_v
                    observations.append(((d.qpos[q:q+7].copy(), d.qvel[v:v+6].copy()), c))
            if observe_stability:
                stability = self._stable(observations)
            else:
                c = self.sensors.settle_support(d)
        else:
            mj.mj_step(m, d, nstep=self.interval)
            c = self.sensors.settle_support(d)
        self.check(begin, self.interval)
        return c, stability

    def check(self, begin, steps):
        d = self.d
        if (abs(float(d.time)-begin-steps*self.dt) > 1e-8 or any(w.number for w in d.warning)
                or not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.qacc))):
            raise FloatingPointError("Native numerical warning, nonfinite state or automatic reset")

    def _stable(self, history):
        pose = np.asarray([s[0] for s, _ in history])
        vel = np.asarray([s[1] for s, _ in history])
        counts = np.asarray([c[1] for _, c in history])
        u = np.asarray([c[2] for _, c in history])
        ev, geo = self.evaluation, self.sensors.geo
        ratio = u[:, 1]/geo["apple_weight_N"]
        return bool(np.all((counts[:, 2] > 0) & (counts[:, [0, 1, 3]].sum(axis=1) == 0)
                           & (ratio >= ev["plate_min_support_ratio"])
                           & (u[:, 0]/geo["apple_weight_N"] <= ev["plate_max_other_support_ratio"])
                           & up_to_speed(vel, ev)) and np.all(placed_geometry(pose, geo, ev)[:, 0]))

    def assets(self):
        return {"xml": self.sim.scene_path, "xml_sha256": self.sim.scene_info["scene_sha256"],
                "robot_sha256": sha256(self.sim.robot_path), "mujoco": mj.__version__,
                "base_xyz_wxyz": np.r_[self.sensors.origin, wxyz(self.sensors.base)],
                "joint_names": self.sim.joint_names, "actuator_indices": self.sim.actuator_ids,
                "actuation": self.sim.scene_info["actuation"], "reset_sha256": sha256(self.path/"scene/reset.json")}

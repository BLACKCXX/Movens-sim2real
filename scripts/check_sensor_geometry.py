"""Check RGB-D coordinate and inference boundaries on a rendered physical reset."""
import argparse
from contextlib import closing
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from simulation.stage4_videos import Views

from movens_sim2real.contract import ROOT, candidate, config, read, rotation, save_arrays, sha256, write
from movens_sim2real.data import array_dict
from movens_sim2real.hardware import AppleMask, DepthTrack
from movens_sim2real.object_info import physical, prepare, quantity
from movens_sim2real.rgbd import DepthObservations, recorded_depth


def run(source,out,worker):
    r=read(source/"summary.json")
    z=array_dict(r["path"])
    kin=r["identity"]["asset"]
    init=array_dict(source/"initial.npz")
    c=candidate(config(),r["identity"]["episode"])
    with closing(Views(dict(candidate=c,scene_xml=kin["xml"]))) as v:
        v.synchronize(init["qpos"],init["qvel"],init["actuator"])
        if worker:
            rgb=v.image("ego")
        v.renderer.enable_segmentation_rendering()
        ids=v.image()
        v.renderer.disable_segmentation_rendering()
        v.renderer.enable_depth_rendering()
        depth=v.image()
        v.renderer.disable_depth_rendering()
        m,d=v.model,v.data
        cam=m.camera("approx_ego_camera").id
        h,w=depth.shape
        f=.5*h/np.tan(np.deg2rad(m.cam_fovy[cam])*.5)
        K=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]])
        base=np.asarray(kin["base_xyz_wxyz"])
        inv=rotation(base[3:]).inv()
        t=np.eye(4)
        t[:3,:3]=inv.as_matrix()@d.cam_xmat[cam].reshape(3,3)@np.diag([1.,-1.,-1.])
        t[:3,3]=inv.apply(d.cam_xpos[cam]-base[:3])
        apple={m.body("apple_root").id}
        for i,j in enumerate(m.body_parentid):
            if j in apple:
                apple.add(i)
        mask=(ids[:,:,1]==mj.mjtObj.mjOBJ_GEOM)&np.isin(ids[:,:,0],np.flatnonzero(np.isin(m.geom_bodyid,list(apple))))
    dims=z["dims"][0]
    if worker:
        out.parent.mkdir(parents=True,exist_ok=True)
        image,arrays,receipt=(out.with_suffix(x) for x in (".png",".npz",".depth.json"))
        Image.fromarray(rgb).save(image)
        save_arrays(arrays,rgb_index=np.array([0]),captured_s=np.array([0.]),available_s=np.array([.04]),depth_m=depth[None])
        write(receipt,dict(format="fixed-base-registered-rgbd-v1",domain="mujoco",depth_unit="m",
            registration="rectified_ego_rgb",clock_source="native MuJoCo render; logical 40 ms delivery for the check",
            calibration_source="compiled MuJoCo optical camera",rgb_sha256=sha256(image),arrays=arrays.name,
            arrays_sha256=sha256(arrays),intrinsics=K,base_from_optical=t))
        meta,a=recorded_depth(receipt,image,np.array([0.]),rgb.shape[:2],domain="mujoco")
        marker=AppleMask()(np.asarray(Image.open(image)),0.)
        adapter=DepthObservations(meta["intrinsics"],meta["base_from_optical"],float(max(dims)))
        adapter.publish(marker,a["depth_m"][0],a["captured_s"][0],a["available_s"][0])
        if adapter.read(.039) is not None or adapter.read(.151) is not None or adapter.read(.04) is None:
            raise ValueError("RGB-D result arrived early, outlived its capture, or was unavailable when due")
        r=adapter.read(.04)[0]
        segmentation=dict(source=marker["source"],pixels=int(marker["mask"].sum()),
            mask_iou=float((mask&marker["mask"]).sum()/(mask|marker["mask"]).sum()),
            early_and_stale_rejected=True,receipt=str(receipt))
    else:
        track=DepthTrack(K,t,float(max(dims)))
        r=track(depth,dict(captured_s=0.,mask=mask,confidence=.6),0.)
        segmentation=None
    if r is None or r["orientation_valid"] or r["linear_valid"]:
        raise ValueError("Depth should yield a center and partial points, not a single-view orientation or velocity")
    error=float(np.linalg.norm(r["pose"][:3]-z["object"][0,:3]))
    if error>.025 or not np.any(r["geometry"]["point_valid"]):
        raise ValueError(f"Optical-to-base point geometry differs: {error:.3f} m")
    priors=prepare("apple",ROOT/"configs/object_sources.json",
                   quantity(dims,[.9*dims,1.1*dims],"m","MuJoCo collision extents for this interface check",.9))
    p=physical(priors)
    if not np.all(p[1,4:7]) or np.any(p[1,7:16]) or p[1,1:4].any():
        raise ValueError("Expected inertia diagonal and missing COM/cross-terms/friction")
    write(out,dict(source=str(source),test_domain="MuJoCo camera rendering, no real depth supplied",
        visible_depth_pixels=int(mask.sum()),center_error_m=error,inferred_orientation=False,
        valid_partial_points=int(r["geometry"]["point_valid"].sum()),segmentation=segmentation,priors=priors))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--recorded-worker",action="store_true",help="Use RGB detection and the recorded depth/arrival adapter")
    args=p.parse_args()
    run(args.source,args.out,args.recorded_worker)

"""Calibrated depth geometry, shared by rendered and recorded sensor inputs."""
import numpy as np

from .contract import rotation
from .features import point_context


def unproject(depth_m, mask, K, base_from_camera, *, min_depth=.08, max_depth=3.):
    d=np.asarray(depth_m)
    K=np.asarray(K,dtype=float)
    transform=np.asarray(base_from_camera,dtype=float)
    rot=transform[:3,:3]
    if (d.ndim!=2 or np.shape(mask)!=d.shape or K.shape!=(3,3) or transform.shape!=(4,4)
        or not np.isfinite(K).all() or not np.isfinite(transform).all()
        or not np.allclose(K[2],[0,0,1]) or min(K[0,0],K[1,1])<=0.
        or not np.allclose(rot.T@rot,np.eye(3),atol=1e-6) or np.linalg.det(rot)<=0.
        or not np.allclose(transform[3],[0,0,0,1])):
        raise ValueError("Expected m and a calibrated right-handed optical camera transform")
    valid=np.asarray(mask,dtype=bool) & np.isfinite(d) & (d>=min_depth) & (d<=max_depth)
    v,u=np.where(valid)
    z=d[v,u]
    camera=(np.column_stack((u,v,np.ones(len(u))))@np.linalg.inv(K).T)*z[:,None]
    return camera@rot.T+transform[:3,3]


def partial_points(base_cloud, pose, confidence, count=256):
    xyz=np.asarray(base_cloud)
    n=min(len(xyz),count)
    result=np.zeros((count,4),dtype=np.float32)
    valid=np.zeros(count,bool)
    cf=np.zeros(count,np.float32)
    if n:
        sample=xyz[np.linspace(0,len(xyz)-1,n,dtype=int)]
        result[:n,:3]=rotation(pose[3:]).inv().apply(sample-pose[:3])
        # Contact permission is unavailable from depth or a segmented mask.
        # Leave its one-channel value neutral; mask/conf describe geometry.
        result[:n,3]=0.
        valid[:n]=True
        cf[:n]=confidence
    return point_context(result,valid,cf)


def empty_cloud():
    return point_context(np.zeros((256,4)),np.zeros(256,bool))

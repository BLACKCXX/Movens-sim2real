"""Residual actor, independent critics and a probabilistic dynamics ensemble."""
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import ACTOR_SIZE, DYN_SIZE, PHYS_SIZE, PRED_SIZE


ARCH = "gru128-geom64-apple-v1"


def mlp(*dims):
    layers = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a,b))
        if i < len(dims)-2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class Norm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))

    def set_stats(self, stat):
        mean, std = stat
        self.mean.copy_(torch.as_tensor(mean))
        self.std.copy_(torch.as_tensor(std))

    def values(self, x):
        return (x-self.mean)/self.std

    def forward(self, vf):
        val, mask, confidence = vf.unbind(dim=-2)
        val = self.values(val).clamp(-15,15)*mask
        return torch.cat((val, mask, confidence), dim=-1)


class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = mlp(4,32,64,64)

    def forward(self, batch):
        x = batch["points"]
        mask = batch["point_valid"]
        cf = batch["point_conf"]*mask
        # SI xyz scaled to ~1 for a 10 cm task; confident geometry comes
        # from actual surfaces or calibrated depth, with an independent mask.
        y = self.points(torch.cat((x[..., :3]*10, x[..., 3:]), dim=-1))
        any_valid = mask.any(dim=-1, keepdim=True)
        pooled = y.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        pooled = torch.where(any_valid, pooled, 0.)
        confidence = cf.amax(dim=-1, keepdim=True).expand_as(pooled)
        return torch.cat((pooled, any_valid.expand_as(pooled), confidence), dim=-1)


class History(nn.Module):
    def __init__(self, width, geom_in_history):
        super().__init__()
        self.norm = Norm(width)
        self.geo = Geometry()
        self.geom_in_history = geom_in_history
        self.gru = nn.GRU(width*3+(192 if geom_in_history else 0), 128, batch_first=True)

    def forward(self, batch):
        g = self.geo(batch)
        x = self.norm(batch["history"])
        if self.geom_in_history:
            x = torch.cat((x,g[:, None].expand(-1,x.shape[1],-1)), dim=-1)
        h, _ = self.gru(x)
        return h[torch.arange(len(h), device=h.device), batch["length"]-1], g


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = History(ACTOR_SIZE, True)
        self.trunk = mlp(128,256,128)
        self.mu = nn.Linear(128,7)
        self.log_std = nn.Linear(128,7)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -2.3)
        self.tracking = nn.Linear(128,43)  # SONIC state-only auxiliary

    def forward(self, batch):
        h, _ = self.encoder(batch)
        f = F.silu(self.trunk(h))
        return self.mu(f), self.log_std(f).clamp(-5,1)

    def sample(self, batch, deterministic=False, generator=None):
        mu, ls = self(batch)
        if deterministic:
            return mu.tanh(), None
        e = torch.randn(mu.shape, device=mu.device, generator=generator)
        u = mu+ls.exp()*e
        z = u.tanh()
        logp = (-.5*e.square()-ls-.5*np.log(2*np.pi)-2*(np.log(2.)-u-F.softplus(-2*u))).sum(-1)
        return z, logp

    def track(self, batch):
        return self.tracking(self.encoder(batch)[0])


class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = History(ACTOR_SIZE, True)
        self.phys = Norm(PHYS_SIZE)
        self.head = mlp(128+PHYS_SIZE*3+7,256,256,1)

    def forward(self,batch,z):
        h, _ = self.encoder(batch)
        return self.head(torch.cat((h,self.phys(batch["physical"]),z), dim=-1)).squeeze(-1)


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = History(DYN_SIZE, False)
        self.phys = Norm(PHYS_SIZE)
        self.action = Norm(43)
        self.target = Norm(PRED_SIZE)
        self.head = mlp(128+43+PHYS_SIZE*3+192+1,256,256,PRED_SIZE*2+5)
        self.tracking = nn.Linear(128,43)

    def forward(self,batch):
        h,g = self.encoder(batch)
        u = self.action.values(batch["action"]).clamp(-15,15)
        y = self.head(torch.cat((h,u,self.phys(batch["physical"]),g,batch["dt"]/.02), dim=-1))
        return y[:, :PRED_SIZE], y[:, PRED_SIZE:PRED_SIZE*2].clamp(-10,2), y[:, -5:]

    def track(self,batch):
        return self.tracking(self.encoder(batch)[0])

    def predict(self,batch):
        mu,lv,e = self(batch)
        return mu*self.target.std+self.target.mean, lv.exp()*self.target.std.square(), e.sigmoid()

    def loss(self,batch):
        mu,lv,e = self(batch)
        valid = batch["delta_valid"]
        evalid = batch["events_valid"]
        d = self.target.values(batch["delta"])
        loss = .5*((lv+(d-mu).square()*(-lv).exp())*valid).sum()/valid.sum().clamp_min(1)
        eloss = (F.binary_cross_entropy_with_logits(e,batch["events"],reduction="none")*evalid).sum()/evalid.sum().clamp_min(1)
        return loss + .2*eloss


def stats(arrays):
    total = count = squared = 0
    for a in arrays:
        x,v,_ = np.moveaxis(a.astype(np.float64), -2, 0)
        total = total + (x*v).sum(axis=0)
        squared = squared + (x*x*v).sum(axis=0)
        count = count + v.sum(axis=0)
    n = np.maximum(count,1)
    mean = total/n
    std = np.sqrt(np.maximum(squared/n-mean*mean,0.))
    # Constant and unknown channels do not require artificial large gains.
    std = np.where((std>1e-6) & (count>1), std, 1.)
    return mean.astype(np.float32),std.astype(np.float32)


def tensors(batch,device):
    if isinstance(batch, Mapping):
        return {k:tensors(v,device) for k,v in batch.items()}
    a = np.asarray(batch)
    d = torch.float32 if np.issubdtype(a.dtype,np.floating) else None
    return torch.as_tensor(a,device=device,dtype=d)


def gather(samples,device):
    def collate(values):
        first = values[0]
        if isinstance(first, Mapping):
            return {k: collate([v[k] for v in values]) for k in first}
        return tensors(values, device)
    return collate(samples)


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_name(path.name+".partial")
    torch.save(state,temp)
    temp.replace(path)


def load_models(path, device="cpu"):
    x = torch.load(path,map_location="cpu",weights_only=False)
    if x["arch"] != ARCH:
        raise ValueError(f"{path}: incompatible model architecture")
    actor = Actor().to(device)
    actor.load_state_dict(x["actor"])
    models = [Dynamics().to(device) for _ in x["dynamics"]]
    for model, weights in zip(models, x["dynamics"]):
        model.load_state_dict(weights)
    return actor.eval(), [m.eval() for m in models], x

"""Prepare cited physical priors before the task; consumers only read the result."""
import argparse
import json
import os
import re
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from .contract import ROOT, config, read, sha256, write
from .features import packed
from .phase import RULE_VERSION


APPLE_USDA_URL = "https://api.nal.usda.gov/fdc/v1/food/171688"


def usda_apple_card(response):
    """Read the cited portions from USDA's SR Legacy raw apple record."""
    if (response.get("fdcId") != 171688 or response.get("dataType") != "SR Legacy"
            or not response.get("description", "").startswith("Apples, raw, with skin")):
        raise ValueError("Expected USDA FDC 171688 (raw apples with skin)")
    portions = {}
    for row in response["foodPortions"]:
        size = re.fullmatch(r'(small|medium|large) \((\d+)(?:-(\d+)/(\d+))?" dia\)', row["modifier"])
        if size:
            label, whole, num, den = size.groups()
            if label in portions or row["amount"] != 1 or (den is not None and int(den) == 0):
                raise ValueError(f"Invalid USDA portion for {label} apple")
            inches = int(whole) + (int(num)/int(den) if den else 0.)
            portions[label] = (row["gramWeight"] / 1000, inches * .0254, row["id"])
    order = ("small", "medium", "large")
    if set(portions) != set(order):
        raise ValueError("USDA record lacks small, medium and large whole apple portions")
    masses, diameters, ids = zip(*(portions[k] for k in order))
    return dict(object="apple", aliases=["raw apple", "apple with skin", "苹果", "fruit"],
        mass_kg=masses, diameter_m=diameters, material="raw fruit with skin", source=APPLE_USDA_URL,
        source_detail=(f"FoodData Central SR Legacy 171688, published {response['publicationDate']}, "
            f"portion IDs {', '.join(map(str, ids))}. Reference portions are not measurements of this object."),
        confidence=.35)


def cache_usda_apple(path):
    if path.is_file():
        return
    query = urlencode({"api_key": os.environ.get("USDA_FDC_API_KEY", "DEMO_KEY")})
    try:
        with urlopen(f"{APPLE_USDA_URL}?{query}", timeout=8) as response:
            item = json.load(response)
    except (URLError, TimeoutError) as exc:
        reason = getattr(exc, "code", None) or type(exc).__name__
        raise ValueError(f"USDA FDC 171688 query failed ({reason}); supply a cached response or the cited offline catalog") from None
    usda_apple_card(item)
    write(path, item)


def words(text):
    return set(re.findall(r"[\w]+", text.casefold()))


def lookup(records, name, diameter=None):
    """Rank similarly named objects, using metric size only when measured."""
    query = words(name)
    matches = []
    for r in records:
        terms = words(" ".join([r["object"], *r["aliases"]]))
        score = len(query & terms) / max(1,len(query))
        if score:
            m,d = np.asarray(r["mass_kg"]),np.asarray(r["diameter_m"])
            if m.shape != d.shape or m.ndim != 1 or np.any(~np.isfinite(np.r_[m,d])) or np.any(m <= 0) or np.any(d <= 0):
                raise ValueError(f"Invalid cited masses/sizes: {r['source']}")
            k = len(m)//2 if diameter is None else int(np.argmin(abs(d-diameter)))
            distance = 0. if diameter is None else abs(np.log(d[k]/diameter))
            matches.append((score*np.exp(-distance),float(m[k]),r))
    return sorted(matches,key=lambda x:(-x[0],x[2]["source"]))


def quantity(value, bounds, unit, source, confidence):
    x,b = np.asarray(value,dtype=float),np.asarray(bounds,dtype=float)
    if (b.shape != (2,*x.shape) or not np.isfinite(b).all() or not np.isfinite(x).all()
            or np.any(x<b[0]) or np.any(x>b[1]) or not source or not 0<confidence<=1):
        raise ValueError(f"Invalid physical prior {unit}: {source}")
    return dict(value=x.tolist(),range=b.tolist(),unit=unit,source=source,confidence=float(confidence),valid=True)


def unpack(e,unit,shape):
    if e["unit"] != unit or np.shape(e["value"]) != shape:
        raise ValueError(f"Wrong prior units or shape: expected {unit} {shape}")
    return quantity(e["value"],e["range"],unit,e["source"],e["confidence"])


def prepare(name,catalog,geometry=None,measurement=None):
    # Optional geometry is a measured/model diameter triplet and uncertainty,
    # in object axes. Incomplete views cannot claim enclosing dimensions.
    dims = unpack(geometry,"m",(3,)) if geometry else None
    if dims and min(dims["range"][0]) <= 0:
        raise ValueError("Metric object dimensions must be strictly positive")
    cards=read(catalog)
    if isinstance(cards,dict):
        cards=[usda_apple_card(cards)]
    matches=lookup(cards,name,max(dims["value"]) if dims else None)
    mass=unpack(measurement["mass"],"kg",()) if measurement and "mass" in measurement else None
    if mass is None and matches:
        confidence,guess,r=matches[0]
        mass=quantity(guess,[min(r["mass_kg"]),max(r["mass_kg"])],"kg",r["source"],
                      r["confidence"]*min(1.,confidence))
        mass["kind"]="similar_portion"
        mass["source_detail"]=r["source_detail"]
    elif mass:
        mass["kind"]="supplied_measurement"
    if mass and mass["range"][0] <= 0:
        raise ValueError("Mass must be strictly positive")
    inertia=None
    if mass and dims:
        def ellipsoid(m,d):
            a=np.square(np.asarray(d)/2)
            return m/5*(a.sum()-a)
        inertia=quantity(ellipsoid(mass["value"],dims["value"]),
             [ellipsoid(mass["range"][j],dims["range"][j]) for j in (0,1)],
             "kg m^2",[mass["source"],dims["source"]],.6*min(mass["confidence"],dims["confidence"]))
        inertia.update(frame="object axes about estimated center of mass",
                       method="uniform ellipsoid; only diagonal estimated, cross terms and COM unknown")
    friction={}
    for pair in ("hand_object","object_support"):
        e=(measurement or {}).get("friction",{}).get(pair)
        friction[pair]=unpack(e,"1,m,m",(3,)) if e else None
        if e and min(friction[pair]["range"][0])<0:
            raise ValueError("Negative contact friction")
    return dict(schema="physical-priors-v1",query=name,catalog_sha256=sha256(catalog),mass=mass,dimensions=dims,
                center_of_mass=None,inertia_diagonal=inertia,
                material=dict(value=matches[0][2]["material"],source=matches[0][2]["source"],
                              confidence=matches[0][0]*matches[0][2]["confidence"],kind="similar_object") if matches else None,
                friction=friction,
                matches=[dict(object=r["object"],score=float(s),source=r["source"]) for s,_,r in matches])


def physical(priors, *, base_gravity=None, base_plane=None, delay_s=None, gains=None):
    """Pack only supplied estimates into the deployed 29D schema."""
    v=np.empty(29)
    v.fill(np.nan)
    mask=np.zeros(29,bool)
    c=np.zeros(29)
    estimates=((0,priors["mass"]), (slice(4,7),priors["inertia_diagonal"]),
        (slice(10,13),priors["friction"]["hand_object"]), (slice(13,16),priors["friction"]["object_support"]),
        (slice(16,19),base_gravity), (slice(19,22),priors["dimensions"]), (slice(22,26),base_plane),
        (26,delay_s), (slice(27,29),gains))
    for sl,e in estimates:
        if e is not None:
            v[sl],mask[sl],c[sl]=e["value"],True,e["confidence"]
    return packed(v,mask,c)


def prepare_task(request,catalog,geometry=None,measurement=None):
    """Freeze an Agent's proposed tool choices against the fixed task protocol."""
    cfg=config()
    if (request.get("task")!=cfg["task"] or request.get("hand")!="left" or request.get("target")!="plate"
            or request.get("phase_rules",cfg["phase"])!=cfg["phase"]):
        raise ValueError("Agent request changed the fixed task, hand, target or phase thresholds")
    name=request["object"]
    if not isinstance(name,str) or not words(name)&{"apple","苹果"}:
        raise ValueError("Agent request does not identify the AppleToPlate object")
    tracker=request["tracker"]
    if tracker not in ("sam3","red_component_tracker"):
        raise ValueError("Unknown perception tool for this task")
    return dict(schema="apple-task-tools-v1",task=cfg["task"],hand="left",target="plate",object=name,
                rule_version=RULE_VERSION,phase_rules=cfg["phase"],tracker=tracker,
                priors=prepare(name,catalog,geometry,measurement))


def read_task(path,cfg):
    task=read(path)
    if (task["schema"]!="apple-task-tools-v1" or task["task"]!=cfg["task"] or task["hand"]!="left"
            or task["target"]!="plate" or task["rule_version"]!=RULE_VERSION or task["phase_rules"]!=cfg["phase"]
            or task["tracker"] not in ("sam3","red_component_tracker") or task["priors"]["schema"]!="physical-priors-v1"):
        raise ValueError(f"{path}: incompatible frozen perception and stage setup")
    return task


def main():
    p=argparse.ArgumentParser(description=__doc__)
    kind=p.add_mutually_exclusive_group(required=True)
    kind.add_argument("--name")
    kind.add_argument("--request",type=Path,help="Prepare a frozen tool setup from an Agent's structured task request")
    origin=p.add_mutually_exclusive_group()
    origin.add_argument("--catalog",type=Path,help="Cited offline cards or a saved USDA record")
    origin.add_argument("--apple-fdc-cache",type=Path,help="Read a saved source response or fetch FDC 171688 once")
    p.add_argument("--geometry",type=Path)
    p.add_argument("--measurement",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    geometry=read(a.geometry) if a.geometry else None
    measurement=read(a.measurement) if a.measurement else None
    if a.apple_fdc_cache:
        cache_usda_apple(a.apple_fdc_cache)
    catalog=a.apple_fdc_cache or a.catalog or ROOT/"configs/object_sources.json"
    result=(prepare_task(read(a.request),catalog,geometry,measurement) if a.request
            else prepare(a.name,catalog,geometry,measurement))
    if a.request:
        result.update(request_sha256=sha256(a.request))
    write(a.out,result)


if __name__=="__main__":
    main()

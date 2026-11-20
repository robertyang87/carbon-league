"""Restore the ten earlier enemy-visible moves for already signed public roots."""
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import analyze
from features import vector, NAMES


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def histories():
    plan=json.loads((HERE/"protocol.json").read_bytes())
    book=HERE.parent/"training_cases.json.gz"
    if sha(book)!=plan["source_case_sha256"]:
        raise ValueError("training cases changed")
    games=json.loads(gzip.decompress(book.read_bytes()))["games"]
    entries=[r for r in analyze.read(analyze.INVENTORY)["rows"] if r["original_role"]=="registered_native_training"]
    sources={(r["source"],r["arm"]):r for r in entries}
    if len(entries)!=52 or len(games)!=52 or len(sources)!=52:
        raise ValueError("training eligibility changed")
    seen=set()
    for g in games:
        cfg,actions=analyze.load(sources[g["source"],g["arm"]])
        if analyze.digest(actions)!=g["commands_sha256"]:
            raise ValueError("action tape changed")
        sim=analyze.FastCarbon(cfg).reset(seed=g["seed"])
        if hashlib.sha256(np.asarray(sim.carbon,np.uint8).tobytes()).hexdigest()!=g["terrain"]:
            raise ValueError("opening tile digest changed")
        lookup={r["step"]:r for r in g["roots"]}
        hist=deque(maxlen=11)
        for t,pair in enumerate(actions):
            obs=sim.obs(g["seat"])
            hist.append(obs)
            if t in lookup:
                r=lookup[t]
                if obs!=r["public"] or analyze.digest(obs)!=r["public_sha256"]:
                    raise ValueError(("public view disagrees",g["seed"],t))
                if r["prefix_sha256"]!=analyze.digest(actions[:t]) or analyze.threats(obs)[0]!=r["target"]:
                    raise ValueError("pre-action threat or provenance mismatch")
                key=(g["terrain"],g["seat"],t,r["prefix_sha256"],r["public_sha256"])
                if key in seen:
                    raise ValueError("duplicate public root")
                seen.add(key)
                yield g,r,list(hist)
            sim.step(*pair)
        if not np.allclose(sim.cash,g["official_scores"],atol=1e-6,rtol=0):
            raise ValueError("terminal replay mismatch")
    if len(seen)!=885:
        raise ValueError("did not reconstruct all 885 preregistered warnings")


def build():
    rows=[];features=[]
    for g,root,hist in histories():
        uid=root["target"]["target_uid"]
        x=vector(hist,uid)
        # Only after sealing x read the separate future cash ledger.
        label=root["actual"]
        amount=float(label["target_positive_own_deposit"])
        signed=sum(float(v[3]) for v in label["banking_events"] if v[1]=="own_delivery" and v[2]==1-g["seat"])
        if amount<0 or abs(signed-amount)>1e-7 or not 1<=label["observed_joint_turns"]<=10:
            raise ValueError("target payment disagrees with signed first-trip ledger")
        rows.append(dict(seed=g["seed"],terrain=g["terrain"],seat=g["seat"],step=root["step"],
                         target_uid=uid,prefix_sha256=root["prefix_sha256"],public_sha256=root["public_sha256"],
                         observed_joint_turns=label["observed_joint_turns"],amount=amount))
        features.append(x)
    return np.stack(features),rows

"""Reproducibly append the registered reversible collection route to the original v1 single file."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

D=Path(__file__).resolve().parent
ROOT=D.parents[3]
PARENT="9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
RULES="9a762c4143b73785667a260b3264738bfa90e5a37894fdf7be6e7b50dad23731"
PROTOCOL="1d6069ed2af8a3afb7df2c3d826ff50fe43031b02519ac32ef9a3ca0b4139c1f"
WRAP='''
import copy as _harvest_copy
import types as _harvest_types
_HARVEST = _harvest_types.ModuleType("registered_public_harvest")
exec(zlib.decompress(base64.b85decode(_HARVEST_SOURCE)),_HARVEST.__dict__)
_HARVEST_V1 = agent
_HARVEST_HISTORY = {}
_HARVEST_PLAN = None
_HARVEST_ISSUED = False
_HARVEST_LAST = None
_HARVEST_TRACE = None
_HARVEST_CFG = {"size":15,"episodeSteps":300,"moveCost":0,"smeltCost":0,"treeLifespan":50,
                "collectorAbsorptionRate":.1,"initialCollectRate":.3,"collectDecreaseRate":0}

def _harvest_remember(obs):
    # Called after inference; retain only fields needed by a future public decision.
    s,t = obs["player"],obs["step"]
    view={"step":t,"player":s,"carbon":list(obs["carbon"]),
          "players":[[None,None,{u:tuple(w) for u,w in p[2].items()}] for p in obs["players"]]}
    _HARVEST_HISTORY[t]=view
    _HARVEST_HISTORY.pop(t-2,None)

def agent(observation,configuration=None):
    global _HARVEST_PLAN,_HARVEST_ISSUED,_HARVEST_LAST,_HARVEST_TRACE
    s,t=observation["player"],observation["step"]
    if t==0 or _HARVEST_LAST!=(s,t-1):
        _HARVEST_PLAN=None
        _HARVEST_ISSUED=False
        _HARVEST_HISTORY.clear()
    base=_HARVEST_V1(observation,configuration)
    cfg={key:(configuration or {}).get(key,default) for key,default in _HARVEST_CFG.items()}
    command, _HARVEST_PLAN, meta = _HARVEST.choose(
        observation, _HARVEST_HISTORY, base, cfg, _HARVEST_PLAN, allow_start=not _HARVEST_ISSUED)
    if meta["status"]=="start":
        _HARVEST_ISSUED=True
    _HARVEST_TRACE=(t,s,dict(base),meta)
    _HARVEST_LAST=(s,t)
    _harvest_remember(observation)
    return command
'''

def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    parent=(ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py").read_bytes()
    policy=(D/"policy.py").read_bytes()
    if (sha(parent),sha(policy),sha((D/"protocol.json").read_bytes()))!=(PARENT,RULES,PROTOCOL):
        raise ValueError("source or protocol identity changed")
    payload=parent+b"\n_HARVEST_SOURCE="+repr(base64.b85encode(zlib.compress(policy,9))).encode()+b"\n"+WRAP.encode()
    if len(payload)>10_000_000:raise ValueError("single file too large")
    compile(payload,"candidate.py","exec")
    manifest={"candidate_sha256":sha(payload),"bytes":len(payload),"parent_sha256":PARENT,"rules_sha256":RULES,
              "protocol_sha256":PROTOCOL,"build_sha256":sha(Path(__file__).read_bytes()),
              "mission":"one mission per game, replanned every tick until its worker banks, dies or aborts"}
    for name,content in (("candidate.py",payload),("package.json",(json.dumps(manifest,indent=2)+"\n").encode())):
        target=D/name
        if target.exists() and target.read_bytes()!=content:raise ValueError("changed existing package")
        if not target.exists():target.write_bytes(content)
    print(manifest)

if __name__=="__main__":main()

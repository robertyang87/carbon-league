import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STUDY=ROOT/"reports/sapling_1200k/master6_actor_dev/strategic_destination_learning"
sys.path.insert(0,str(STUDY))
import free_entry
PROTOCOL=json.loads((HERE/"protocol.json").read_bytes())

def sha(s): return hashlib.sha256(s).hexdigest()
def need(ok,m):
    if not ok:raise ValueError(m)
def original():
    src=subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:sources/master3s2/master3s2.py"])
    need(sha(src)==PROTOCOL["source_master_sha256"],"Examiner changed")
    return src
def module(src):
    m=types.ModuleType("local_original_m3s2")
    exec(compile(src,"local_original_m3s2.py","exec"),m.__dict__)
    return m
def build(params):
    baseline=original()
    marker=b"def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    source=(HERE/"runtime.py").read_text()
    lines=["_OPP_H = {"]
    for k,arr in params.items():
        a=np.asarray(arr,dtype="<f2")
        need(np.isfinite(a).all() and (not k.endswith("scale") or np.all(a>0)),"Invalid readout")
        payload=base64.b64encode(a.tobytes()).decode()
        lines.append(f'    "{k}": np.frombuffer(base64.b64decode("{payload}"),dtype="<f2").reshape({a.shape!r}).astype(np.float32),')
    lines.extend(["}"])
    addition=source.replace("# PARAMETER_TABLE","\n".join(lines)).replace("\n","\r\n").encode()
    call=b"    lwo, lbo, _ = net(xo.astype(np.float32), _SID)\r\n"
    newcall=b"    lwo, lbo = _opponent_root_readout(xo, lwo, lbo, players, op)\r\n"
    need(baseline.count(marker)==baseline.count(call)==1,"Unexpected root site")
    blob=baseline.replace(marker,addition+marker).replace(call,call+newcall)
    need(blob.replace(addition,b"").replace(newcall,b"")==baseline,"Changed other native behavior")
    return blob

def tape(c):
    source=ROOT/c["path"]
    raw=source.read_bytes() if source.exists() else subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:"+c["path"]])
    need(sha(raw)==c["sha256"],"Altered source tape")
    t=json.loads(gzip.decompress(raw))
    if "initial_terrain_u8_sha256" in t:
        terrain=t["initial_terrain_u8_sha256"]
    else:
        import random
        random.seed(c["seed"]);np.random.seed(c["seed"])
        env=free_entry.official_rules(ROOT)("carbon",configuration={**t["game_configuration"],"randomSeed":c["seed"]})
        terrain=sha(bytes(int(v) for v in env.states[0].observation.full_carbon))
    need(t["examiner_md5"]==PROTOCOL["opponent_hash"] and terrain==c["terrain"] and
         t["candidate_seat"]==c["candidate_seat"] and not t["errors"],"Wrong physical opponent")
    return t

def stream(c,model):
    historical=tape(c);seat=c["candidate_seat"];op=1-seat
    tracker=model.BeliefTracker(model._Bel,seat)
    prev=None
    # Only seat's own public observation is passed to the belief or the neural input.
    signed_card={**c,"expert_seat":seat}
    for t,obs,cfg,own in free_entry.public_history(signed_card):
        players=obs["players"]
        est,known=tracker.step(players,t,obs["carbon"])
        joint=historical["actions"][t]
        need(own==joint[seat] and obs["player"]==seat and "full_carbon" not in obs,"Wrong seat or action chronology")
        if t in PROTOCOL["steps"]:
            xo=model._quant(model.encode(players,t,op,est,[1.]*model.N,prev))[None]
            yield t,obs,cfg,xo,own,joint[op]
        tracker.set_actions(players,own);prev=players

def dump(path,value):
    data=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    if path.suffix==".gz":data=gzip.compress(data,mtime=0)
    if path.exists() and data!=path.read_bytes(): raise ValueError("Frozen output differs")
    path.write_bytes(data)

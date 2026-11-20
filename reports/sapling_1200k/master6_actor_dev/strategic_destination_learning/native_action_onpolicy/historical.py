"""Register a real, action-independent Monte Carlo control variate for the upcoming games."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from build import HERE, sha

INTAKE=HERE.parent/"native_action_return_intake"
sys.path.insert(0,str(INTAKE))
import select_orders as sel

COMMON=sel.common

def main():
    root=COMMON.REPO
    fit=INTAKE/"continuation_value"/"fit.json"
    records=json.loads(fit.read_bytes())["source_games"]
    if len(records)!=20 or len(set(c["terrain"] for c in records))!=10:
        raise ValueError("Source contract")
    rows=[]
    for ref in records:
        path=root/ref["archive"]
        if sha(path)!=ref["archive_sha256"]:
            raise ValueError("Historical source changed")
        with zipfile.ZipFile(path) as z:
            commands=z.read(ref["arm"]+"/turns.jsonl")
            reg=json.loads(z.read(ref["arm"]+"/registration.json"))
            receipt=json.loads(z.read(ref["arm"]+"/receipt.json"))
        if hashlib.sha256(commands).hexdigest()!=ref["turns"] or not receipt["counted"]:
            raise ValueError("Historical command receipt changed")
        cfg={**COMMON.collect.CFG,"actTimeout":1,"agentTimeout":20,"runTimeout":1500}
        env=COMMON.collect.begin(ref["seed"],cfg)
        seat=ref["seat"]
        if hashlib.sha256(bytes(env.states[0].observation.full_carbon)).hexdigest()!=ref["terrain"]:
            raise ValueError("Historical terrain changed")
        if reg["files"]["candidate"]["sha256"]!=COMMON.SIGNED["bc"] or reg["files"]["opponent"]["sha256"]!=COMMON.SIGNED["original"]:
            raise ValueError("Historical players changed")
        cash=[]
        for t,l in enumerate(commands.splitlines()):
            row=json.loads(l)
            if t!=row["step"] or any(r["status"]!="ok" for r in row["calls"]): raise ValueError("Historical per-step failure")
            if t%6==0:
                p=COMMON.collect.public(env,seat)["players"]
                cash.append(float(p[seat][0])-float(p[1-seat][0]))
            env.step([r["response"]["action"] for r in row["calls"]],[])
        scores=[player.reward for player in env.states]
        if not env.done or scores!=receipt["official_scores_by_seat"] or len(cash)!=50:
            raise ValueError("Historical terminal mismatch")
        margin=float(scores[seat]-scores[1-seat])
        rows.append(dict(seed=ref["seed"],seat=seat,terrain=ref["terrain"],source_sha256=ref["archive_sha256"],
                         official_margin=margin,remaining=[round(margin-c,2) for c in cash]))
    out=HERE/"baseline.json"
    text=json.dumps(dict(source_fit=sha(fit),times=list(range(0,295,6)),games=rows),sort_keys=True,indent=2)+"\n"
    if out.exists() and out.read_text()!=text: raise ValueError("Prior baseline changed")
    out.write_text(text)

if __name__=="__main__":
    main()

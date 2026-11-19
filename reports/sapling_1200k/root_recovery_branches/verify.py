"""Independently replay each stored arm and calculate the tagged tree's budget-relative effects."""
import gzip
import json
from pathlib import Path

from branches import DIRS, OUT, begin, digest


def replay(raw,seat,root,tail):
    env=begin(raw)
    start=root["t"];t=start;tid=root["tree"];pref=raw["actions"][:start]
    for step in pref:env.step(step)
    revenue=[0.,0.];owned=[0,0];owners={}
    first_seizure=None
    for i,act in enumerate(tail,1):
        if env.done:raise ValueError("data outlast official terminal")
        env.step(act); t+=1
        state=env.states[0].observation
        side=next((s for s,p in enumerate(state.players) if tid in p[3]),None)
        if side is not None:
            owned[side]+=1
            revenue[side]+=state.trees[tid][1]
        if i in (1,2,3,5,10,20):owners[str(i)]=side
        if side==seat and first_seizure is None:
            first_seizure=t
    if not env.done:raise ValueError("branch ends early")
    return dict(score=[x.reward for x in env.states],end_t=t,owned=owned,income=[round(x,4) for x in revenue],
                owners_after=owners,first_focal_ownership_state=first_seizure)


def main():
    core=json.loads((OUT/"results.json").read_bytes())
    if not core["complete"]:raise ValueError("branches not complete")
    output=[]
    for g in core["games"]:
        seed,seat=g["seed"],g["seat"]
        path=DIRS[seed]/f"raw_{seed}_{seat}.json.gz"
        b=path.read_bytes()
        if digest(b)!=g["source_sha256"]:raise ValueError("bad root file")
        raw=json.loads(gzip.decompress(b))
        cert=json.loads((DIRS[seed]/"official_replay.json").read_bytes())["runs"][DIRS[seed].name]["games"]
        raw["initial_terrain_u8_sha256"]=next(x["map_sha256"] for x in cert if (x["seed"],x["seat"])==(seed,seat))
        for family,r in g["roots"].items():
            rb=(OUT/r["branch_tape"]).read_bytes()
            if digest(rb)!=r["branch_tape_sha256"]:raise ValueError("changed tape")
            root=r["root"];tail=json.loads(gzip.decompress(rb))
            a=replay(raw,seat,root,raw["actions"][root["t"]:])
            b=replay(raw,seat,root,tail)
            if a["score"]!=g["baseline_score"] or b["score"]!=r["official_scores"]:
                raise ValueError("independent interpreter result mismatch")
            def dif(f):return round(b[f][seat]-a[f][seat],4)
            output.append(dict(seed=seed,seat=seat,family=family,t=root["t"],uid=root["worker"],tid=root["tree"],
                               margin_delta=r["margin_delta"],delta_u=r["delta_u"],target_own_income_delta=dif("income"),
                               target_own_tenure_delta=dif("owned"),target_opponent_income_delta=round(b["income"][1-seat]-a["income"][1-seat],4),
                               baseline=a,counterfactual=b))
    (OUT/"verification.json").write_text(json.dumps(output,ensure_ascii=False,allow_nan=False,indent=2)+"\n")


if __name__=="__main__":main()

"""Probe public greedy+guard calls at every stratified tick; controls do not start matches."""
import argparse
from collections import Counter
import gzip
import json
from types import ModuleType

from pydantic_core import from_json

import action_parity as audit
import actions
import build
import policy

HERE = build.HERE


def create(raw,name):
    module = ModuleType(name)
    exec(compile(raw,name,"exec"),module.__dict__)
    return module


def fixture(value):
    width = len(actions.FEATURES) + len(actions.KINDS)
    return {"features":list(actions.FEATURES),"kinds":list(actions.KINDS),"enabled":True,
            "heads":{k:([value if k=="margin" else 0.] + [0.] * (width-1)) for k in policy.TARGETS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned",action="store_true")
    args = parser.parse_args()
    if args.learned:
        params = {"learned":json.loads((HERE/"weights.json").read_bytes())}
    else:
        params = {"positive":fixture(16.),"negative":fixture(0.)}
    data = (HERE/"action_parity.json.gz").read_bytes()
    parity = from_json(gzip.decompress(data))
    audit.check(parity["action_sha256"]==build.sha((HERE/"actions.py").read_bytes()),"parity coverage identity")
    reference, _, _, defaults = audit.get_actor()
    actors = {}
    receipts = {}
    for key,p in params.items():
        binary, receipt = build.compose(p)
        if args.learned:
            persisted = json.loads((HERE/"package.json").read_bytes())
            audit.check(persisted["candidate_sha256"]==build.sha(binary) and binary==(HERE/"candidate.py").read_bytes()
                        and persisted["weights_sha256"]==build.sha((HERE/"weights.json").read_bytes()),"rerender identity")
            receipt = persisted
        actors[key] = create(binary,"probe_"+key)
        receipts[key] = receipt
    detail = []
    counts = {k:Counter() for k in params}
    for card in parity["maps"]:
        seat = card["seat"]
        blob = (audit.REPLAYS/(card["replay"]+".json.gz")).read_bytes()
        audit.check(build.sha(blob)==card["source_sha256"],"source")
        game = from_json(gzip.decompress(blob))
        cfg={**defaults,**(game.get("configuration") or {})}
        previous = {k:False for k in params}
        decisions=[]
        taken={k:0 for k in params}
        reference_roots={r["t"]:r for r in card["roots"]}
        for t,frame in enumerate(game["steps"][:297]):
            view={k:frame[0]["observation"][k] for k in ("step","trees","players")}
            view.update(player=seat,carbon=frame[seat]["observation"]["carbon"])
            if t not in (0,254,296) and not 255<=t<=295:
                for actor in [reference,*actors.values()]:
                    actor._FEAT.observe(view)
                continue
            baseline=reference.agent(view,cfg)
            legal=[]
            if t in reference_roots:
                root=reference_roots[t]
                audit.check(root["parent"]==baseline,("frozen parent",t))
                allowed={(u,original,a,k) for u,original,a,k in root["valid"]}
                legal=[p for p in actions.proposals(view,cfg,baseline)
                       if (p["uid"],p["original"],p["action"],p["kind"]) in allowed]
            for key,actor in actors.items():
                actual=actor.agent({**view,"full_carbon":[99999]*225,"reveal":{"14":9999.0},"other_side_private_logit":-17.},cfg)
                trace=actor._VALUE_TRACE
                audit.check(trace["parent"]==baseline,("parent",key,t))
                expected=policy.choose(legal,params[key]) if not previous[key] else None
                if expected:
                    uid=expected["uid"]
                    order=expected["action"]
                    altered=dict(baseline)
                    if order=="NONE":
                        altered.pop(uid,None)
                    else:
                        altered[uid]=order
                    audit.check(trace["chosen"]==(uid,order,expected["kind"]) and altered==actual and actual!=baseline,
                                ("one-slot positive",key,t))
                    previous[key]=True
                    taken[key]+=1
                else:
                    audit.check(actual==baseline and trace["chosen"] is None,("false positive",key,t))
                menu=trace["offered"]
                audit.check(len(menu)<=4 and len({p[2] for p in menu})==len(menu),("menu",key,t))
                if menu:
                    first={}
                    for p in legal:
                        first.setdefault(p["kind"],(p["uid"],p["action"],p["kind"]))
                    audit.check(menu==list(first.values()),("representatives",key,t))
                counts[key][trace["reason"]]+=1
                decisions.append([key,t,trace["reason"],menu,trace["chosen"]])
        for k in taken:
            audit.check(taken[k]<=1 and (k!="positive" or taken[k]==1) and (k!="negative" or taken[k]==0),
                        ("once-per-game",k,card["replay"]))
            audit.check(actors[k]._COUNTS==reference._COUNTS,"real parent network prefix contributions")
        detail.append({"replay":card["replay"],"terrain":card["terrain"],"seat":seat,"taken":taken,
                       "decisions":decisions})
    result={"mode":"learned" if args.learned else "constant diagnostic coefficients","packages":receipts,
            "parity_sha256":build.sha(data),"probe_sha256":build.sha(open(__file__,"rb").read()),
            "maps":detail,"reasons":{k:dict(v) for k,v in counts.items()}}
    target="package_parity.json.gz" if args.learned else "package_control.json.gz"
    build.atomic(HERE/target,gzip.compress((json.dumps(result,separators=(",",":"),allow_nan=False)+"\n").encode(),mtime=0))
    print(result["reasons"])


if __name__=="__main__":
    main()

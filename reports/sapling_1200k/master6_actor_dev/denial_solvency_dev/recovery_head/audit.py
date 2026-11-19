"""Re-audit the largest blind errors from native cash ledgers and probe temporal feature isolation."""
from collections import defaultdict
import copy
import gzip
import json
import math
from dataset import HERE, Game, LATEST, fetch, packed_replay, previous_manifests
from economy_extract import reconstruct, KEYS
from head import features, predict


def main():
    audit=fetch.load(HERE/"evaluation.json")
    rows=[json.loads(line) for line in gzip.decompress((HERE/"events.jsonl.gz").read_bytes()).splitlines()]
    cards={g["id"]:g for p in previous_manifests(LATEST) for g in fetch.load(p)["games"]}
    examples=[]
    for item in audit["largest_absolute_errors"][:3]:
        gid=item["game"]; t=item["pre_t"]; s=item["seat"]; rival=1-s
        raw=json.loads(gzip.decompress(packed_replay(cards[gid])))
        game=Game(raw); cfg=raw["configuration"]; window=range(max(0,t-10),t+1)
        public=features([game._obs(u,s) for u in window],s,cfg)
        if public!=item["public"]:raise ValueError("saved example features differ")
        clone=copy.deepcopy(raw)
        clone["steps"]=clone["steps"][:t+2]
        clone["rewards"]=[None,None]
        for frame in clone["steps"]:
            frame[0]["observation"]["full_carbon"]=[999999]
            frame[1-s]["observation"]["carbon"]=[-123456]
            for side in frame: side["action"]={"fiction":"UP"}
        for side in clone["steps"][-1]:side["observation"]={"erased_future":True}
        blank=Game(clone)
        if features([blank._obs(u,s) for u in window],s,cfg)!=public:
            raise ValueError("future, actual commands or opponent private information changed features")
        flow=reconstruct(raw["steps"],cfg)[0][t:t+30]
        cash={name:round(sum(line[rival][j] for line in flow),4) for j,name in enumerate(KEYS)}
        states=[]
        for k in range(t,t+31):
            other=game._obs(k,s,terminal=True)["players"][rival]
            states.append({"t":k,"cash":other[0],"trees":len(other[3]),"collectors":sum(w[2]=="COLLECTOR" for w in other[2].values()),
                           "planters":sum(w[2]=="PLANTER" for w in other[2].values())})
        if abs(sum(cash.values())-(states[-1]["cash"]-states[0]["cash"]))>.001:raise ValueError("cash channels not reconciled")
        if int(not any(p["collectors"] for p in states[1:]))!=item["label_30"]:raise ValueError("label")
        recruits=[]
        for k in range(t,t+30):
            pre=game._obs(k,s)["players"][rival]
            actual=raw["steps"][k+1][rival].get("action") or {}
            for base in pre[1]:
                if actual.get(base) in ("RECCOLLECTOR","RECPLANTER"):
                    recruits.append({"t":k,"base":base,"command":actual[base],"pre_cash":pre[0]})
        examples.append({"recruit_requests_in_30":recruits,"gid":gid,"actor":item["actor"],"rival":item["rival"],"pre_t":t,"p":item["predictions"]["state"],"target":item["label_30"],
                         "cash_flow_next30":cash,"states":states})
    # Independent equal-map calculations using the one-file stdlib deployment path only.
    val=[r for r in rows if r["split"]!="train" and r["label_30"] is not None]
    weights=fetch.load(HERE/"weights.json")
    by=defaultdict(list)
    for r in val:by[r["map"]].append((r["label_30"],predict(r["public"],weights["state"])))
    brier=sum(sum((y-p)**2 for y,p in rs)/len(rs) for rs in by.values())/len(by)
    nll=sum(sum(-math.log(p if y else 1-p) for y,p in rs)/len(rs) for rs in by.values())/len(by)
    original=audit["groups"]["main"]["models"]["state"]
    if abs(brier-original["Brier"])>1e-12 or abs(nll-original["log_loss"])>1e-12:
        raise ValueError("map-equal deployed inference differs")
    fetch.dump(HERE/"audit.json",{"deployment_brier":brier,"deployment_logloss":nll,
                                 "probes_change_hidden_carbon_future_commands_and_scores":len(examples),"examples":examples})


if __name__=="__main__":main()

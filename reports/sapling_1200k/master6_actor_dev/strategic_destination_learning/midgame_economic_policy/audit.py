"""Independently recover sample weights, future bank movements and frozen public team changes."""
from collections import Counter,defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from pydantic_core import from_json

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
CACHE=Path("/workspace/sapling-all-top5-20261119/replays")


def read(p):
    raw=p.read_bytes()
    return from_json(gzip.decompress(raw) if p.suffix==".gz" else raw)


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def main():
    rules=read(HERE/"protocol.json"); labels=read(HERE/"advantage.json.gz"); held=read(HERE/"held_commands.json.gz")
    fits=read(HERE/"fit.json"); gate=read(HERE/"entry.json")
    excluded=set()
    for name,hash_ in rules["exclude_hashes"].items():
        p=STUDY/"opponent_matched_economy"/name
        assert sha(p)==hash_;excluded.update(r["meta"]["terrain"] for r in read(p))
    train_maps={r["terrain"] for r in labels};held_maps={r["terrain"] for r in held}
    assert len(train_maps)==234 and len(held_maps)==38 and not train_maps&held_maps and not (train_maps|held_maps)&excluded
    panel=STUDY.parent/"broader_elite_backbone_dev/panel.json";assert sha(panel)==rules["panel"]
    cards={r["id"]:r for r in read(panel)["games"]}
    by_game=defaultdict(list)
    for r in [*labels,*held]:by_game[r["gid"]].append(r)
    assert len(by_game)==272
    for key,rows in by_game.items():
        card=cards[key];p=CACHE/(key+".json.gz");assert sha(p)==card["sha256"]
        replay=read(p);seat=card["seat"]
        for r in rows:
            assert (card["terrain_sha256"],seat,card["split"])==(r["terrain"],r["seat"],r["split"])
            v=[replay["steps"][t][0]["observation"]["players"] for t in (r["t"],r["t"]+20)]
            delta=(v[1][seat][0]-v[0][seat][0])-(v[1][1-seat][0]-v[0][1-seat][0])
            assert abs(delta-r["target"])<1e-7
        assert [r["t"] for r in rows]==[t for t in rules["times"] if t+20<=len(replay["steps"])-1]
    for case,group in by_game.items():
        if group[0]["split"]=="train":
            assert abs(sum(r["weight"] for r in group)-len(group))<1e-9
            assert len({r["fold"] for r in group})==1
    max_error=0
    for fold in range(5):
        train=[r for r in labels if r["fold"]!=fold]
        test=[r for r in labels if r["fold"]==fold]
        assert not {r["terrain"] for r in train}&{r["terrain"] for r in test}
        cnt=Counter(r["terrain"] for r in train);w=np.array([1/cnt[r["terrain"]] for r in train]);wn=w/w.sum()
        x=np.array([r["regressors"] for r in train]);y=np.array([r["target"] for r in train])
        avg=wn@x;sd=np.sqrt(wn@((x-avg)**2)).clip(min=1e-6); z=(x-avg)/sd
        design=np.vstack([np.sqrt(w)[:,None]*z,np.sqrt(rules["ridge"])*np.eye(z.shape[1])])
        responses=np.hstack([np.sqrt(w)*(y-wn@y),np.zeros(z.shape[1])])
        beta=np.linalg.lstsq(design,responses,rcond=None)[0]
        for r in test:
            actual=wn@y+((r["regressors"]-avg)/sd)@beta
            max_error=max(max_error,abs(actual-r["baseline"]))
    assert max_error<1e-6
    assert len({fits["arms"][a]["package"] for a in ("uniform","weighted")})==2 and set(gate["export_sha256"])=={"parent","uniform","weighted"}
    for arm in ("uniform","weighted"):
        assert sha(HERE/(arm+".py"))==fits["arms"][arm]["package"]==gate["export_sha256"][arm]
        assert sha(HERE/(arm+".npz"))==fits["arms"][arm]["weights"]
    diff=Counter();roles=Counter(); cases=[]
    for r in held:
        a=r["uniform"]["actions"];b=r["weighted"]["actions"]
        if a!=b:
            diff[r["split"]]+=1; diff[str(r["seat"])]+=1
            if len(cases)<4:cases.append(dict(gid=r["gid"],seat=r["seat"],t=r["t"],opportunity=r["opportunity"],
                                               actual_bank_target=r["target"],uniform=a,weighted=b,expert=r["expert"]))
        for uid,role in r["roles"].items():
            roles[role]+=a.get(uid,"NONE")!=b.get(uid,"NONE")
    changed=sum(diff.get(s,0) for s in ("val","holdout"))
    assert changed==gate["pooled"]["changed"]["uniform_vs_weighted"]
    output=dict(verified_release_sha256=gate["export_sha256"],train_rows=len(labels),held_rows=len(held),training_maps=len(train_maps),
        held_maps=len(held_maps),zero_physical_overlap_with_studied_maps=True,active_full_team_differences=changed,
        differences_by_split_and_physical_seat=diff,changed_entities_by_role=roles,first_changed_cases=cases,
        max_crossfit_lstsq_error=max_error,held_log_sha256=sha(HERE/"held_commands.json.gz"),protocol=sha(HERE/"protocol.json"))
    (HERE/"audit.json").write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


if __name__=="__main__":
    main()

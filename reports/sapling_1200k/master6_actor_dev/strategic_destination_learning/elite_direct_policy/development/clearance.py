"""Reproduce the physical terrain and public-replay exclusion for the four elite-BC development maps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import types

D = Path(__file__).resolve().parent
STRAT = D.parents[1]
SAP = STRAT.parents[1]
ROOT = SAP.parents[1]
LEAGUE = STRAT.parent/"cargo_tree_action_learning_dev/native_actor_corpus/league_policy"
SOURCE = "54134b9b5d1dc991eb510223cfde1548653a089e"
BASE = "99ec3fed8f48158b85399f6887bf62036205dabe"
SEEDS = list(range(621758520,621758524))
TOP5 = SAP/"all_top5/master3s2_expert_orders/manifest.json"


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


engine=module("census", LEAGUE/"clearance.py")
git=engine.git
sha=engine.sha


def frozen(path):
    return git("show",SOURCE+":"+path.relative_to(ROOT).as_posix())


def boards():
    out={}
    for s in SEEDS:
        engine.np.random.seed(s)
        obs=engine.make("carbon",configuration={**engine.exam.CFG,"randomSeed":s}).states[0].observation
        h=engine.map_identity.terrain_sha256(obs.full_carbon)
        if h!=engine.map_identity.local_board(s):
            raise ValueError(f"Official/FastCarbon opening mismatch on {s}")
        out[str(s)]=h
    if len(set(out.values()))!=len(SEEDS):
        raise ValueError("Within-panel physical map collision")
    return out


def top_ledgers(paths):
    known,proof,faults={}, {}, []
    for path in paths:
        if "/all_top5/" not in path or not path.endswith("/validation_maps.jsonl.gz"):
            continue
        raw=git("show",SOURCE+":"+path)
        proof[path]=sha(raw)
        for line in gzip.decompress(raw).splitlines():
            r=json.loads(line)
            if r["id"] in known:
                raise ValueError("duplicated canonical game")
            known[r["id"]]=(r["seed"],r["initial_terrain_u8_sha256"])
            if "/unscored/" in path:
                faults.append(r["id"])
    if len(faults)!=1 or len(known)<1648:
        raise ValueError("incomplete canonical physical-ledger population")
    return known,proof,faults


def census():
    paths=git("ls-tree","-r","--name-only",SOURCE).decode().splitlines()
    top5,proof,_=top_ledgers(paths)
    template=LEAGUE/"clearance.py"
    original=frozen(template)
    adapted=original.decode()
    for old,new in (("len(top5)==1637",f"len(top5)=={len(top5)}"),
                    ("if b'6217584' in b and CANDIDATE.search(b)","if CANDIDATE.search(b)")):
        if adapted.count(old)!=1:
            raise ValueError("historical scanner changed")
        adapted=adapted.replace(old,new)
    m=types.ModuleType("frozen_elite_census")
    m.__file__=str(template)
    exec(compile(adapted,str(template),"exec"),m.__dict__)
    m.SNAPSHOT,m.SEEDS,m.GROUPS=SOURCE,SEEDS,{"fresh_development":SEEDS}
    m.CANDIDATE=re.compile(rb"(?<![.\d])62175852[0-3](?![.\d])")
    out=m.scan()
    physical=out["by_seed"]
    if physical!=boards():
        raise ValueError("inconsistent generated boards")
    protocol=D.parent/"protocol.json"
    leader=json.loads(frozen(protocol))
    raw=frozen(TOP5)
    if sha(raw)!=leader["source_manifest_sha256"]:
        raise ValueError("different expert-order inventory")
    roles={r["id"]:k for k,entries in leader["entries"].items() for r in entries}
    groups={"fit":[],"assessment":[],"protected":[]}
    for r in json.loads(raw)["games"]:
        category=roles.get(r["id"])
        if not category:
            if r["disposition"]!="protected":
                raise ValueError("unaccounted original-opponent map")
            category="protected"
        groups[category].append({k:r[k] for k in ("id","seed","terrain")})
    if {k:len(rr) for k,rr in groups.items()}!={"fit":5,"assessment":6,"protected":19}:
        raise ValueError("incomplete elite/BC exclusion")
    for rows in groups.values():
        for r in rows:
            if r["seed"] in SEEDS or r["terrain"] in physical.values() or tuple(top5[r["id"]])!=(r["seed"],r["terrain"]):
                raise ValueError("elite-map overlap or inconsistent physical ledger")
    previous=STRAT/"replication/clearance.json"
    old=json.loads(frozen(previous))
    if set(old["by_seed"]) & set(physical) or set(old["by_seed"].values()) & set(physical.values()):
        raise ValueError("repeat-development map reused")
    return dict(out,schema="sapling-elite-four-map-development/clearance/1",seeds=SEEDS,
                purpose="Development only for the frozen elite-finetuned Sapling and immediate-action BC. Permanently excluded from training and the independent formal exam.",
                physical_boards=physical,elite_maps_excluded=groups,elite_protocol_sha256=sha(frozen(protocol)),
                expert_manifest_sha256=sha(raw),previous_replication_boards_sha256=sha(frozen(previous)),
                checked_current_canonical_ledgers=proof,census_template_sha256=sha(original))


def score_delta():
    registry=module("fresh_score_sources",LEAGUE/"natural_original/fresh_repeat/clearance.py")
    registry.SCORE_BASE,registry.SNAPSHOT=BASE,SOURCE
    cards,info=registry.score_census()
    paths=git("ls-tree","-r","--name-only",SOURCE).decode().splitlines()
    known,proof,faults=top_ledgers(paths)
    previous=[]
    def remember(k,s,h):
        if known.get(k,(s,h))!=(s,h):
            raise ValueError("conflicting replay identity "+k)
        known[k]=(s,h)
    for path, key in ((STRAT/"development/clearance.json","score_replay_openings"),
                      (STRAT/"development/clearance_increment.json","replays"),
                      (STRAT/"replication/score_delta.json","additional_public_replays")):
        data=frozen(path)
        rows=json.loads(data)[key]
        for r in rows:
            remember(r["game"],r["seed"],r["terrain"])
        previous.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(data),rows=len(rows)))
    index={}
    for p in paths:
        name=Path(p).name
        if name.endswith(".json.gz") and name[:-8] in cards:
            index.setdefault(name[:-8],[]).append(p)
    cache=Path(tempfile.gettempdir())/"sapling-elite-clearance-replays"
    cache.mkdir(exist_ok=True)
    def obtain(k):
        card=cards[k]
        url=f"{registry.fetch.SERVER}/replay/{card['id']}/{card['game']}"
        local=cache/(k+".bin")
        def network():
            b=registry.fetch.Retrieval({},0.2).get(url)
            registry.fetch.atomic(local,b)
            return b
        options=[(p,lambda p=p:git("show",SOURCE+":"+p)) for p in index.get(k,[])]
        if local.is_file():
            options.append((url,local.read_bytes))
        options.append((url,network))
        for src,get in options:
            try:
                body=get()
                env,_,_=registry.fetch.extract(body)
                v=registry.fetch.validate(env,card)
                h=engine.map_identity.terrain_sha256(env["steps"][0][0]["observation"]["full_carbon"])
                return dict(game=k,seed=v["random_seed"],terrain=h,source=src,body_sha256=sha(body),
                            terminal=v["statuses"],turns=v["last_step"],scores_by_seat=v["rewards_by_seat"])
            except Exception:
                if src==options[-1][0] and get==options[-1][1]:
                    raise
        raise ValueError("exhausted sources for "+k)
    with ThreadPoolExecutor(4) as pool:
        rows=list(pool.map(obtain,sorted(cards)))
    for r in rows:
        remember(r["game"],r["seed"],r["terrain"])
    if set(info["first_seen_top5"])-set(known):
        raise ValueError("unreconciled top-five addition")
    physical=boards()
    if any(s in SEEDS or h in physical.values() for s,h in known.values()):
        raise ValueError("replay overlaps a candidate board")
    return dict(schema="sapling-elite-four-map-development/score-clearance/1",base=BASE,through=SOURCE,
                physical_boards=physical,seed_intersections=[],physical_intersections=[],canonical_top5_ledgers=proof,
                canonical_top5_faults=faults,prior_score_receipts=previous,additional_public_replays=rows,
                score_delta=info,reconciled_source_ids=len(known),
                reconciled_identity_sha256=sha(json.dumps(sorted((k,*v) for k,v in known.items()),separators=(",",":")).encode()),
                unresolved_incremental_replays=[])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("census","scores"))
    mode=parser.parse_args().mode
    p=D/("clearance.json" if mode=="census" else "score_delta.json")
    value=census() if mode=="census" else score_delta()
    raw=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
    if p.exists() and p.read_bytes()!=raw:
        raise ValueError("attempt to overwrite different registration")
    tmp=p.with_suffix(".new")
    tmp.write_bytes(raw)
    tmp.replace(p)
    print(p.name,sha(raw),flush=True)

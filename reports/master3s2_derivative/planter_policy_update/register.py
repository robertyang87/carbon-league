"""Select archived DS ladder terrain and opponent-family splits without consulting actions or rewards."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LADDER = HERE.parent / "ladder"
SNAPSHOT = LADDER / "nn_patch_cutoff"
FORMAL = (1_887_000_000,1_888_000_000)
HELD_FAMILIES = ("fable","master","ansheng","doubao")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def family(name):
    norm = unicodedata.normalize("NFKC",name).casefold()
    groups = {"fable":"fable","master":"claude-alphacarbon-master","ansheng":"安生",
              "krypton":"krypton","doubao":"doubao","hajimi":"哈基米","fish":"小肥鱼",
              "one-c":r"^1c-","first":r"^first","ryan":r"^ryanx","stephen":r"^stephen",
              "kimi":r"^kimi","carbonedge":r"^carbonedge","carago":r"^carago",
              "agyc":r"^agycarbon","ljm":r"^ljm","qwen":r"^qwen"}
    for group, pattern in groups.items():
        if re.search(pattern,norm):return group
    return re.sub(r"[-_ ]?v?\d+(\.\d+)*.*$","",norm).strip("-_ .") or norm


def digest_ticket(p):
    raw=p.read_bytes()
    game=json.loads(gzip.decompress(raw))
    cfg=game["configuration"]
    if game["statuses"]!=["DONE","DONE"] or len(game["steps"])!=300 or cfg["episodeSteps"]!=300:return None
    full=game["steps"][0][0]["observation"]["full_carbon"]
    assert len(full)==225 and all(int(x)==x and 0<=x<=255 for x in full)
    terrain=sha(bytes(int(x) for x in full))
    rule_config={k:v for k,v in cfg.items() if k!="randomSeed"}
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(raw),terrain=terrain,replay_seed=cfg["randomSeed"],
                rule_configuration_sha256=sha(json.dumps(rule_config,sort_keys=True,separators=(",",":")).encode()))


def get_candidates():
    available={}
    for p in sorted(LADDER.rglob("*-1.json.gz"),key=lambda p:(p.parent.parent!=SNAPSHOT,str(p))):
        if re.fullmatch(r"[0-9a-f]{32}-1\.json\.gz",p.name):
            available.setdefault(p.name[:32],p)
    out=[];part={}
    for version in ("v1","v2"):
        receipt=json.loads((LADDER/f"ds-pro-{version}.json").read_bytes())
        ident=receipt["platform"]["id"]
        pages=SNAPSHOT/f"{version}_pages.json.gz"
        part[version]=dict(registration=ident,package_sha256=receipt["candidate_sha256"],pages=str(pages.relative_to(ROOT)),sha256=sha(pages.read_bytes()))
        for page in json.loads(gzip.decompress(pages.read_bytes())):
            for m in page["data"]["matches"]:
                path=available.get(m["id"])
                if path is None or m["state"]!="done": continue
                side="a" if ident==m["a"] else "b"
                assert m[side]==ident
                opponent="b" if side=="a" else "a"
                for g in m["result"]:
                    if g["game"]!=1 or g["steps"]!=299 or set(g["statuses"].values())!={"DONE"}:continue
                    replay=digest_ticket(path)
                    if replay is None: continue
                    if any(FORMAL[0]<=seed<FORMAL[1] for seed in (g["seed"],replay["replay_seed"])):continue
                    out.append(replay|dict(version=version,seat=g["order"].index(side),match=m["id"],opponent=m["name_"+opponent],
                              opponent_id=m[opponent],family=family(m["name_"+opponent]),card_seed=g["seed"],
                              card_map_sha256=g["map_sha256"],rule="official 15x15, two bases, episodeSteps=300",
                              source_package_sha256=receipt["candidate_sha256"]))
    return out,part


def select(pool,held):
    def tie(r):
        return sha(("planter-policy-map/1/"+r["terrain"]).encode())
    source=[r for r in pool if (r["family"] in HELD_FAMILIES)==held]
    selected=[];terrains=set();opps=collections.Counter();seats=collections.Counter();versions=collections.Counter();fam=collections.Counter();late=collections.Counter()
    for i in range(8 if held else 20):
        eligible=[r for r in source if r["terrain"] not in terrains and (not held or r["family"]==HELD_FAMILIES[i%4])]
        def priority(r):
            cutoff=int("nn_patch_cutoff/replays/" in r["path"])
            return ((0 if held else fam[r["family"]]),opps[r["opponent_id"]],versions[r["version"]],
                     seats[r["seat"]],late[cutoff],tie(r))
        choice=min(eligible,key=priority)
        selected.append(dict(split="held" if held else "fit",**choice))
        terrains.add(choice["terrain"]);opps[choice["opponent_id"]]+=1;fam[choice["family"]]+=1
        versions[choice["version"]]+=1;seats[choice["seat"]]+=1;late[int("nn_patch_cutoff/replays/" in choice["path"])]+=1
    return selected


def main():
    pool,bind=get_candidates()
    fit,held=select(pool,False),select(pool,True)
    for key in ("terrain","opponent_id","family","card_map_sha256"):
        assert not {r[key] for r in fit}&{r[key] for r in held},key
    for s in (fit,held):
        assert set(r["seat"] for r in s)=={0,1} and set(r["version"] for r in s)=={"v1","v2"}
        assert len(set(r["terrain"] for r in s))==len(s) and any("nn_patch_cutoff" in r["path"] for r in s)
    def count(s):
        return dict(maps=len(s),registrations=len({x["opponent_id"] for x in s}),families=dict(collections.Counter(x["family"] for x in s)),
                    by_seat=dict(collections.Counter(x["seat"] for x in s)),by_version=dict(collections.Counter(x["version"] for x in s)),
                    from_latest_window=sum("nn_patch_cutoff" in x["path"] for x in s))
    result=dict(schema="ds-planter-first-tenure/1",source=bind,tickets=fit+held,counts={k:count(s) for k,s in (("fit",fit),("held",held))},
          selection="Full matching local public replay; never read score, winner, action, or later-frame boards to select. Fixed opponent families, seats/version balance, deterministic terrain hash tie. Actual seat is the index of the strategy side in result.order. Family heuristic based on public registration names cannot certify shared hidden code.",
          isolation={"reserved_seed_half_open":FORMAL,"source":"reports/sapling_1200k/tree_win_gate.py:37","physical_id":"SHA-256 of raw uint8 full_carbon at t0; used only for dedup and split","exclude_from_any_future_final":True},
          labels="Only actual own planter actions; every observed successful plant or seizure plus same UID prior 1 and 2 actions. Nearest future event wins on overlap. Add neutral own actions at t % 9 == 4 when no attributed event. Include decision t0 if qualified. Cash target = actual first consecutive tenure rent minus actual plant/seize cost; later recapture gets separate episode. Negative cash means downweight actual action without fabricating an alternative label.",
          weighting="w = exp(clip(net_cash/100,-1,1)*exp(-lag/2)); BC w=1. Normalize map weights. Source V2 96-D own root trunk; same-capacity action heads. CE + 0.25 KL(parent||candidate) + .003/2 L2. Bounded tanh incremental logits of amplitude 1.25.",
          pass_criteria="Map-equal, event-weighted NLL vs matched BC and zero V2. On positive realized first-tenure targets mean logp of executed action >= BC; on negative targets mean logp of executed action < BC. Report separate denominators and each terrain/opponent. Only actual sampled actions. No strength assertion from imitation or root action change.",
          mechanism="For each held terrain take exactly the two lowest SHA256(f'planter-native/{terrain}/{step}') across t>=24 and t<=298 roots with >=1 active own planter, excluding none by previous or current orders, rent, outcome, or whether the action changed. If <2, keep all; 16 max. Preserve base/candidate physical command, zero changes & time/receipt in native .58s search. No full games.",
          rule_config_hash_note="SHA of sorted compact JSON configuration excluding only randomSeed")
    raw=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    dest=HERE/'protocol.json'
    if dest.exists():assert dest.read_bytes()==raw
    else:dest.write_bytes(raw)
    print(sha(raw),result['counts'])

if __name__=='__main__':main()

"""Build immutable same-state action/official-return pairs from existing complete native Master3s2 continuations."""
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE / "root_recovery_branches"))
from branches import begin, clean, public
from tree_win_gate import validate_examiner_configuration

V1 = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
EXAMINER = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
ORIGINAL_MD5 = "57f9a594cce4f2ec9e3573d8d9d26ca1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def sha(x):
    return hashlib.sha256(x).hexdigest()


def checksum(p):
    return sha(p.read_bytes())


def rel(p):
    return p.relative_to(ROOT).as_posix()


def read(p):
    data = p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix == ".gz" else data)


@lru_cache(maxsize=64)
def source(p, expected_sha):
    require(checksum(p) == expected_sha, f"changed source {p}")
    raw = read(p)
    require(raw["examiner_md5"] == ORIGINAL_MD5 and len(raw["actions"]) == raw["steps"]
            and not raw["errors"] and not raw["audit_messages"], "incomplete or different original source")
    validate_examiner_configuration(raw["configuration"])
    return raw


def arm(tape, scores, statuses, calls, receipt_file, action_file):
    require(statuses == ["DONE", "DONE"] and len(tape) > 0 and calls == len(tape)-1, f"incomplete live response {receipt_file}")
    return {"tape": tape, "scores": scores, "calls": calls,
            "receipt": rel(receipt_file), "receipt_sha256": checksum(receipt_file),
            "action_file": rel(action_file), "action_sha256": checksum(action_file)}


def live(folder):
    prot = read(folder/"live_protocol.json")
    records = {}
    name2 = "alternative" if (folder/"live_alternative.json").is_file() else "capture"
    for armname, filename in (("parent", "parent"), ("alternative", name2)):
        path = folder/f"live_{filename}.json"
        item = read(path); tape_file = folder/item["action_tape"]; tape = read(tape_file)
        require(item["arm"] == filename and item["source_sha256"] == EXAMINER
                and item["protocol_sha256"] == checksum(folder/"live_protocol.json")
                and item["shared_init_sha256"] == checksum(folder/"live_shared_init.json")
                and item["action_sha256"] == checksum(tape_file) and tape[0] == item["root_actions"]
                and item["terminal_t"] == prot["root_t"] + len(tape), f"live identity {path}")
        records[armname] = (item, arm(tape, item["score"], item["statuses"], item["search_calls"], path, tape_file))
    parent, alt = records["parent"][0], records["alternative"][0]
    for key in ("seed", "seat", "root", "pack_sha256", "original_trajectory_sha256", "root_opponent_observation_sha256"):
        require(parent[key] == alt[key], f"live branches differ at {key}, {folder}")
    if not prot.get("record", {}).get("source"):
        require(folder == BASE/"root_recovery_branches" and prot["seed"] == 621454920, f"no original tape {folder}")
        raw = BASE/"terminal_economy/original_parent"/f"raw_{parent['seed']}_{parent['seat']}.json.gz"
    else:
        raw = ROOT/prot["record"]["source"]
    require(read(folder/"live_shared_init.json")["source_sha256"]==EXAMINER, "wrong response stream")
    return dict(folder=folder, family=prot["family"], raw=raw, raw_sha=parent["original_trajectory_sha256"], seed=parent["seed"],
                seat=parent["seat"], t=parent["root"]["t"], pack=parent["pack_sha256"], rng=parent["shared_init_sha256"],
                repeated="repeatability" in folder.as_posix(), arms={k:v[1] for k,v in records.items()})


def stratified(folder, card, common_protocol):
    pair = read(folder/"pair.json")
    require(common_protocol["parent_sha256"]==V1 and pair["id"]==card["id"], "stratified registration")
    arms={}; streams=[]
    for armname, fname, expected in (("parent","base",card["base"]),("alternative","alt",card["alt"])):
        af=folder/f"{fname}.json.gz"; data=read(af); b=data["binding"]; history=data["history"]
        require(pair["files"][fname]==checksum(af) and b["card_sha256"]==pair["card_sha256"]
                and b["frozen_sha256"]==pair["frozen_sha256"] and b["id"]==card["id"] and b["arm"]==fname
                and data["joint0"][card["seat"]]==expected and data["joint0"][1-card["seat"]]==card["root_enemy"]
                and [h["t"] for h in history]==list(range(card["t"]+1,data["terminal_step"]))
                and all(h["issue"]["binding"]==b and h["issue"]["t"]==h["t"] for h in history)
                and data["original_config"]["_TEMP"]==1. and data["original_config"]["_SEED"] is None
                and data["original_config"]["_SEARCH_OK"] and data["original_config"]["_RX"], "stratified receipt")
        streams.append(b["paired_seed"])
        arms[armname]=arm([data["joint0"],*[h["joint"] for h in history]],data["official_scores"],data["statuses"],
                           data["completed_original_calls"],af,af)
    require(streams[0]==streams[1], "different stratified stream")
    return dict(folder=folder, family="stratified_"+card["proposal"]["kind"], raw=ROOT/card["raw"], raw_sha=card["raw_sha256"],
                seed=card["seed"],seat=card["seat"],t=card["t"],pack=V1,rng=streams[0],repeated=card["stream"]!=0,arms=arms)


def residual(folder):
    arms={};streams=[]
    for name,fname in (("parent","base"),("alternative","alt")):
        af=folder/f"{fname}.json.gz";d=read(af); s=d["source"]
        selected=read(folder.parent/f"selection_{s['seed']}_{s['seat']}.json")
        r=selected["root"]
        require(s==selected["source"] and selected["identities"]["original"]==EXAMINER and selected["identities"]["parent"]==V1
                and d["orders"][0][s["seat"]]==r[fname] and d["arm"]==fname
                and [v["step"] for v in d["search"]]==list(range(r["t"]+1,d["final_step"]))
                and d["original_config"]["_TEMP"]==1. and d["original_config"]["_SEED"] is None
                and d["original_config"]["_SEARCH_OK"] and d["original_config"]["_RX"]
                and read(folder/f"{fname}_account.json.gz")["arm_evidence_sha256"]==checksum(af), "residual identity")
        streams.append(d["rng_seed"]);arms[name]=arm(d["orders"],d["official_scores"],d["final_status"],d["original_calls"],af,af)
    require(streams[0]==streams[1], "residual streams differ")
    return dict(folder=folder,family="late_local_residual",raw=ROOT/s["raw"],raw_sha=s["raw_sha256"],seed=s["seed"],
                seat=s["seat"],t=r["t"],pack=V1,rng=streams[0],repeated=False,arms=arms)


def last_turn(folder, case, raw):
    af=folder/f"native_{case['seed']}_{case['seat']}.json.gz";d=read(af);p=read(folder/"native_protocol.json")
    require(p["source_examiner"]==EXAMINER and p["parent_sha256"]==V1 and case["source_sha256"]==d["source_sha256"]
            and d["protocol_sha256"]==checksum(folder/"native_protocol.json"), "terminal search registration")
    result=read(folder/"native_result.json")
    require(checksum(af) in result["pair_sha256"].values(), "unsigned terminal pair")
    arms={}
    for name,key in (("parent","v1"),("alternative","candidate")):
        u=d["arms"][key];tape=[u["first_joint"],u["last_joint"]]
        require(u["search_call_count"]==1 and u["rng_before_sha256"]==d["paired_rng_state_sha256"], "last-step coverage")
        arms[name]=arm(tape,u["end"]["score"],u["end"]["status"],1,af,af)
    return dict(folder=af,family="endgame_queue",raw=raw,raw_sha=d["source_sha256"],seed=d["seed"],seat=d["seat"],
                t=case["t"],pack=V1,rng=d["paired_rng_state_sha256"],repeated=False,arms=arms)


def root_label(spec):
    f,s,t=spec["folder"],spec["seat"],spec["t"]; raw=source(spec["raw"],spec["raw_sha"])
    require((raw["seed"],raw["candidate_seat"],raw["pack_sha256"])==(spec["seed"],s,spec["pack"]),f"source identity {f}")
    require(clean(spec["arms"]["parent"]["tape"][0][s])==clean(raw["actions"][t][s]),f"unexpected parent command {f}")
    require(all(clean(a["tape"][0][1-s])==clean(raw["actions"][t][1-s]) for a in spec["arms"].values()),"opponent root differs")
    env=begin(raw)
    for cmd in raw["actions"][:t]:
        require(not env.done, "prefix ended before root")
        env.step(cmd)
    obs=public(env,s);own=obs["players"][s]
    require(obs["step"]==t,"different observation root")
    pair={k:clean(a["tape"][0][s]) for k,a in spec["arms"].items()}
    changed=sorted(k for k in (pair["parent"].keys()|pair["alternative"].keys()) if pair["parent"].get(k)!=pair["alternative"].get(k))
    require(changed and all(k in own[1] or k in own[2] for k in changed), f"unchanged or unknown root change {f}")
    roles={"base" if k in own[1] else own[2][k][2].lower() for k in changed}
    visible={k:v for k,v in obs.items() if k!="remainingOverageTime"}
    out=dict(seed=spec["seed"],seat=s,t=t,terrain=raw["initial_terrain_u8_sha256"],source=rel(spec["raw"]),source_sha256=spec["raw_sha"],
             parent_sha256=spec["pack"],examiner_sha256=EXAMINER,family=spec["family"],case=rel(f),role=next(iter(roles)) if len(roles)==1 else "mixed",
             response_rng=spec["rng"],changed_uids=changed,public_root_sha256=sha(encode(visible)),root_opponent=clean(raw["actions"][t][1-s]),
             parent=pair["parent"],alternative=pair["alternative"], map_group=raw["initial_terrain_u8_sha256"], arms={})
    # Root identity excludes response stream, end results and path; repeated draws must remain in a single map fold.
    out["identity"]=sha(encode([out["terrain"],s,t,out["public_root_sha256"],pair,out["root_opponent"]]))
    out["repeat_stream"]=spec["repeated"]
    for name,info in spec["arms"].items():
        if name!="parent":
            env=begin(raw)
            for cmd in raw["actions"][:t]:env.step(cmd)
        for j,cmd in enumerate(info["tape"]):
            require(not env.done, f"command after termination {f}")
            env.step(cmd)
            if j==0:
                after=env.states[0].observation.players[s]
                survived={k:k in after[2] for k in changed if k in own[2]}
        scores=[a.reward for a in env.states]
        require(env.done and [a.status for a in env.states]==["DONE","DONE"] and scores==info["scores"]
                and env.states[0].observation.step==t+len(info["tape"]), f"official native branch score {f}")
        out[name+"_scores"]=scores
        out["arms"][name]={k:v for k,v in info.items() if k not in ("tape","scores")}
        out["arms"][name].update(terminal_step=env.states[0].observation.step,next_worker_survival=survived)
    known=f/"training_label.json"
    if known.is_file():
        old=read(known)
        for buywait in ("pay","defer"):
            l=old[buywait];nm=l["native_arm"]
            require(l["scores_by_physical_seat"]==out[nm+"_scores"] and l["actual_action_tape_sha256"]==out["arms"][nm]["action_sha256"],"old debit identity")
            out["arms"][nm].update(signed_focal_uid=old["focal_uid"],signed_focal_root_debit=l["root_debit"],signed_first_tenure_net=l["first_tenure_net"])
    return out


def main():
    specs=[]
    for done in sorted(BASE.rglob("live_parent.json")):
        specs.append(live(done.parent))
    strat=ACTOR/"stratified_action_value_dev"
    cards={c["id"]:c for c in read(strat/"cards.json")["cases"]}
    for file in sorted((strat/"native").glob("*/pair.json")):
        specs.append(stratified(file.parent,cards[file.parent.name],read(strat/"protocol.json")))
    for file in sorted((ACTOR/"late_local_residual_dev/native").glob("*/base.json.gz")):
        specs.append(residual(file.parent))
    terminal=ACTOR/"endgame_queue_dev"
    for case in read(terminal/"native_protocol.json")["cases"]:
        paths=[p for p in (ACTOR/"tree_response_training").rglob(f"raw_{case['seed']}_{case['seat']}.json.gz")
               if checksum(p)==case["source_sha256"]]
        require(len(paths)==1, "ambiguous terminal source")
        specs.append(last_turn(terminal,case,paths[0]))

    primary=[];repeats=[];other=[]
    for i,spec in enumerate(specs):
        if spec["pack"]!=V1:
            row={k:v for k,v in spec.items() if k not in ("folder","raw","arms","repeated")}
            row.update(case=rel(spec["folder"]),source=rel(spec["raw"]),reason="different frozen parent; excluded from v1 fit",
                       arms={k:{kk:vv for kk,vv in v.items() if kk!="tape"} for k,v in spec["arms"].items()})
            other.append(row);continue
        row=root_label(spec)
        (repeats if row["repeat_stream"] else primary).append(row)
        if i%10==0:print("checked",i+1,spec["family"],flush=True)
    prim_ids=[r["identity"] for r in primary];require(len(set(prim_ids))==len(prim_ids), "duplicate original action root")
    require(all(r["identity"] in set(prim_ids) for r in repeats),"no canonical primary for a repeated stream")
    marker=set()
    for r in primary+repeats:
        stream=(r["identity"],r["response_rng"])
        require(stream not in marker, "one action root and identical response RNG repeated")
        marker.add(stream)
    other_t288=ACTOR/"economy_aux_dev/native_dev/paired_edits/native_t288"
    other.append(dict(case=rel(other_t288),reason="economic auxiliary checkpoint follow-up on both arms; different policy from frozen v1",
                      audit=rel(other_t288/"README.md"),audit_sha256=checksum(other_t288/"README.md")))
    data=dict(schema="sapling-v1-native-matched-joint-actions-v1",extractor_sha256=checksum(Path(__file__)),frozen_parent_sha256=V1,
              original_examiner_sha256=EXAMINER,rows=primary,repeated_streams=repeats,other_parent_or_policy=other,
              rows_by_family=dict(Counter(r["family"] for r in primary)),rows_by_role=dict(Counter(r["role"] for r in primary)),
              independent_terrains=len({r["terrain"] for r in primary}),scope="One row per distinct intervention with true new native decisions after its shared committed simultaneous root. Each score array is ordered by physical seat. No formal games. Keep all rows with the same terrain together; selected repeats are segregated from training. A two-unit editing can have one role. Unrun proposals, fixed future tapes, proxy evaluations, deterministic last-turn rollouts and whole-game A/B results are excluded.")
    payload=gzip.compress(encode(data)+b"\n",mtime=0)
    out=HERE/"signed_roots.json.gz"
    if out.exists():require(out.read_bytes()==payload,"published labels changed")
    else:
        tmp=out.with_suffix(".partial");tmp.write_bytes(payload);tmp.replace(out)
    print("signed",len(primary),len(repeats),len(other),data["rows_by_role"],data["independent_terrains"],sha(payload),flush=True)


if __name__=="__main__":
    main()

"""Inventory complete natural original-Master3s2 action tapes; no policy calls or new games."""
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parents[1]
FAMILY = NATIVE.parent
MASTER = FAMILY.parent
SAP = MASTER.parent
ROOT = SAP.parents[1]
REV = "d77348ae4eff2e42bf31717205d6de7d58057237"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
RAW_INVENTORY = "3220787149a5d54b8638d049897e598b0204734c97a44f927b6046fa73de6ae2"

def require(ok, why):
    if not ok:
        raise ValueError(why)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def rel(p):
    return p.relative_to(ROOT).as_posix()

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def moves_key(moves):
    return sha(canonical(moves))

def read(path):
    return json.loads(path.read_bytes())

def load_actions(row):
    """Return the two physical seats' executed commands at each official transition."""
    data = (ROOT/row["source"]).read_bytes()
    require(sha(data) == row["source_sha256"], "Source changed")
    if row["kind"] == "raw":
        return json.loads(gzip.decompress(data))["actions"][:row["n"]]
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        tape = archive.read(row["turns_member"])
    require(sha(tape) == row["turns_sha256"], "Hosted turns changed")
    return [[c["response"]["action"] for c in json.loads(line)["calls"]] for line in tape.splitlines()]

def load_game(row):
    """Return the exact played environment configuration and both seats' executed moves."""
    if row["kind"]=="raw":
        source=(ROOT/row["source"]).read_bytes()
        require(sha(source)==row["source_sha256"],"Changed native raw bytes")
        raw=json.loads(gzip.decompress(source))
        config=raw["game_configuration"]
        moves=raw["actions"][:row["n"]]
    else:
        path=FAMILY/"direct_original/games/candidate/raw_621758346_1.json.gz"
        source=path.read_bytes()
        require(sha(source)=="7a5967383c78324d248f4b228541d7eb30f3ac3d39bf18e74f81a011a78c4631","Changed hosted configuration")
        config=dict(json.loads(gzip.decompress(source))["game_configuration"],
                    randomSeed=row["seed"],agentTimeout=20,actTimeout=1,runTimeout=1500)
        moves=load_actions(row)
    require(len(moves)==row["n"] and moves_key(moves)==row["actions_sha256"],"Different executed command stream")
    return config,moves


def geography(raw):
    short = read(MASTER/"opponent_cargo_calibration_dev/sources.json")["native"]
    lookup = {r["path"]:r for r in raw}
    short_boards = collections.defaultdict(set)
    for game in short:
        other = lookup[game["source"]]
        require((game["sha256"],game["seed"],game["terrain"],game["seat"]) ==
                (other["raw_sha256"],other["seed"],other["physical_sha256"],other["candidate_seat"]),
                "Short-head map conversion differs from official replay")
        if game["split"] in ("train","calibration"):
            short_boards[game["split"]].add(game["terrain"])
    long = read(MASTER/"freight_horizon/dataset.json")["rows"]
    selected = {r["id"]:r for r in read(MASTER/"data_split.json")["games"]}
    maps = {}
    for f in (SAP/"all_top5").rglob("validation_maps.jsonl.gz"):
        for line in gzip.open(f,"rt"):
            r=json.loads(line)
            require(r["id"] not in maps,"Duplicate ladder map identity")
            maps[r["id"]]=r
    freight = set()
    for g in long:
        if g["split"] != "train":
            continue
        s=selected[g["id"]];m=maps[g["id"]]
        require(g["replay_sha256"]==s["sha256"] and s["split"]=="train" and
                (s["seed"],s["terrain_sha256"])==(m["seed"],m["initial_terrain_u8_sha256"]),
                "Long-head source bytes and opening map ledger disagree")
        freight.add(m["initial_terrain_u8_sha256"])
    pins={}
    for p in (HERE.parent/"natural_original/fresh_repeat/clearance.json",HERE.parent/"native_return_intake/clearance.json",
              HERE.parent/"native_return_intake/validation/clearance.json"):
        pins.update({int(k):v for k,v in read(p)["by_seed"].items()})
    require(set(pins)==set(read(HERE/"protocol.json")["excluded_seeds"]) and
            [len(short_boards[s]) for s in ("train","calibration")]==[17,4] and len(freight)==218,
            "Trained map or reservation census changed")
    return {"short_fit":short_boards["train"],"short_calibration":short_boards["calibration"],
            "long_fit":freight,"reserved":set(pins.values())}, pins

def raw_games(inventory):
    for r in inventory:
        good=(r["native_action_supervision"] and r["native_outcome_supervision"] and r["verified"] and
              r["full_official_terminal"] and not r["raw_errors"] and
              r["clock"]["identity"]["receipt_source_sha256"]==ORIGINAL and r["clock"]["configuration"]["natural_t1_058"])
        if not good:
            yield {"source":r["path"],"reason":"No complete clean original-opponent action and official terminal proof"}
            continue
        b=(ROOT/r["path"]).read_bytes()
        require(sha(b)==r["raw_sha256"],"Raw trajectory changed")
        tape=json.loads(gzip.decompress(b)); n=r["transitions"]
        actions=tape["actions"][:n]
        require(len(actions)==n and not any(r["unrecognized_or_unowned_command_ids_by_seat"]) and
                tape["last_post_cash"]==r["terminal_post_cash"] and r["official_score_by_seat"]==tape["official_score_by_seat"] and
                ("initial_full_carbon" not in tape or sha(bytes(map(int,tape["initial_full_carbon"])))==r["physical_sha256"]),"Raw evidence changed")
        yield dict(seed=r["seed"],seat=r["candidate_seat"],expert_seat=r["expert_seat"],terrain=r["physical_sha256"],
                   kind="raw",source=r["path"],arm=None,n=n,terminal_cash=r["terminal_post_cash"],official_scores=r["official_score_by_seat"],
                   source_sha256=r["raw_sha256"],actions_sha256=moves_key(actions),engine_receipt=r["trajectory_sha256"],
                   opponent_sha256=ORIGINAL,candidate_sha256=r["opponent_pack_sha256"])

def hosted_game(path, archive_sha, z, member):
    receipt=json.loads(z.read(member))
    prefix=member[:-len("receipt.json")]
    tape_path=prefix+"turns.jsonl"
    if "official_transitions" not in receipt or receipt.get("status")!="complete" or tape_path not in z.namelist():
        return None
    if not receipt["counted"]:
        return None
    ticket=prefix+"registration.json"; base=FAMILY/"ladder_live/protocol.json"
    if ticket in z.namelist():
        data=z.read(ticket); plan=json.loads(data); arm=plan["arm"]
    elif path==FAMILY/"ladder_live/hosted_attempt/artifact.zip" and prefix=="game/":
        data=base.read_bytes(); plan=json.loads(data); arm="candidate"
    else:
        raise ValueError("No original registration for complete archived game: "+str(path))
    candidate=plan["candidate_seat"]; native=1-candidate
    require(receipt["seed"]==plan["seed"] and receipt["candidate_seat"]==candidate and
            receipt["protocol_sha256"]==sha(data) and plan["files"]["opponent"]["sha256"]==ORIGINAL,
            "Wrong original opponent archive")
    init=receipt["initialization"]
    require(all(c["status"]=="ok" and c["response"]["ready"] for c in init) and init[native]["response"]["sha256"]==ORIGINAL and
            init[candidate]["response"]["sha256"]==plan["files"]["candidate"]["sha256"] and
            init[native]["response"]["original"]==dict(_TEMP=1,_SEED=None,_S_ABORT=.58,_SEARCH_OK=True),
            "Wrong initialized seat or search temperature/budget")
    turns=z.read(tape_path);require(sha(turns)==receipt["turns_sha256"],"Wrong tape")
    sys.path.insert(0,str(FAMILY/"ladder_live")) if str(FAMILY/"ladder_live") not in sys.path else None
    import live
    live.PLAN=plan
    env,_=live.official(); bank=[5.,5.]; actions=[]
    for i,line in enumerate(turns.splitlines()):
        step=json.loads(line)
        require(not env.done and step["step"]==env.states[0].observation.step==i,"Wrong transition")
        cmds=[]
        for s,call in enumerate(step["calls"]):
            obs=live.observe(env,s,bank[s])
            require(sha(live.encoded(obs))==step["public_observation_sha256"][s],"Wrong public state")
            check=dict(call);live.check_action(check,obs);require(check["status"]=="ok","Invalid command")
            cmds.append(check["response"]["action"])
        require(len(cmds)==2,"Missing seat");bank=step["overage_remaining_seconds"]
        env.step(cmds);actions.append(cmds)
    n=len(actions);scores=[s.reward for s in env.states]
    require(env.done and all(s.status=="DONE" for s in env.states) and 0<n==receipt["official_transitions"]<=299 and
            receipt["official_scores_by_seat"]==scores,"Wrong official terminal")
    cash=[p[0] for p in env.states[0].observation.players]
    return dict(seed=plan["seed"],seat=candidate,expert_seat=native,terrain=plan["physical_map_sha256"],
                kind="hosted",source=rel(path),arm=arm,n=n,terminal_cash=cash,official_scores=scores,
                source_sha256=archive_sha,turns_member=tape_path,turns_sha256=sha(turns),engine_receipt=sha(z.read(member)),
                actions_sha256=moves_key(actions),opponent_sha256=ORIGINAL,candidate_sha256=plan["files"]["candidate"]["sha256"])

def build():
    require(sha((ROOT/"data/master3s2_source.py").read_bytes())==ORIGINAL,"Different original")
    paths=subprocess.check_output(["git","-C",str(ROOT),"ls-tree","-rz","--name-only",REV]).decode().split("\0")[:-1]
    inv=NATIVE/"expansion/inventory.jsonl.gz";b=inv.read_bytes();require(sha(b)==RAW_INVENTORY,"Changed exact inventory")
    raw=[json.loads(l) for l in gzip.decompress(b).splitlines()]
    require({r["path"] for r in raw}=={p for p in paths if re.search(r"/raw_\d+_[01]\.json\.gz$",p)},"Unaccounted original raw tape")
    groups,pins=geography(raw);rows=list(raw_games(raw));archives=[]
    for name in paths:
        if not name.startswith(rel(SAP)+"/") or not name.endswith(".zip"):
            continue
        path=ROOT/name
        blob=path.read_bytes() if path.is_file() else subprocess.check_output(["git","-C",str(ROOT),"show",REV+":"+name])
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            recs=[p for p in z.namelist() if p=="receipt.json" or p.endswith("/receipt.json")]
            digest=sha(blob);good=0
            for m in recs:
                got=hosted_game(path,digest,z,m)
                if got is not None:rows.append(got);good+=1
        archives.append(dict(source=name,sha256=digest,receipts=len(recs),complete_original=good))
    admitted=[];removed=[];seen={}
    for r in rows:
        if "reason" in r:removed.append(r);continue
        i=(r["terrain"],r["seat"],r["actions_sha256"])
        r["identifier"]=sha(canonical(i))
        if i in seen:
            removed.append({**r,"reason":"duplicate","primary":seen[i]});continue
        seen[i]=r["source"]
        blockers=[k for k,v in groups.items() if r["terrain"] in v]
        if r["seed"] in pins and "reserved" not in blockers:raise ValueError("Reserved seed disagrees with opening")
        if 1887000000<=r["seed"]<1888000000:blockers.append("formal_exam")
        if blockers:removed.append({**r,"reason":blockers})
        else:admitted.append(r)
    return dict(revision=REV,examiner_sha256=ORIGINAL,raw_inventory_sha256=RAW_INVENTORY,
                physical_signature="SHA256 of 225 row-major unsigned integral true-carbon opening cells",
                seat="physical Sapling/candidate observer; expert_seat indexes the full original Master3s2",
                terminal_cash="Both actual post-transition cash balances in physical seat order, separately from extinction-aware official_scores.",
                reserved_by_seed=pins,excluded_terrain_groups={k:sorted(v) for k,v in groups.items()},
                zip_inventory=archives,unique_complete_actions=len(seen),eligible_games=len(admitted),
                eligible_maps=len({r["terrain"] for r in admitted}),rows=admitted,excluded=removed)

if __name__=="__main__":
    d=build()
    out=HERE/"sources.json"
    text=json.dumps(d,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if out.exists() and out.read_text()!=text:raise ValueError("Source census differs from previously frozen output")
    if not out.exists():out.write_text(text)
    print(d["unique_complete_actions"],d["eligible_games"],d["eligible_maps"],sha(out.read_bytes()))

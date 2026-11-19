"""Optional eight-game development comparison. Starts only after a certified PASS."""
import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
ACTORS = DEV.parent
BASE = ACTORS.parent
ROOT = BASE.parents[1]
OUTPUT = HERE / "whole_games"
PLAN = HERE / "pilot_plan.json"
PARENT = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
LOCK = ACTORS / "tree_response_training/.continuation.lock"
SCANNER = ACTORS / "freight_deployment/map_preflight.py"
CATALOG = SCANNER.parent / "first_response/map_preflight.json"


def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)


def sha(x):
    return hashlib.sha256(x if isinstance(x,bytes) else x.read_bytes()).hexdigest()


def check(v, msg):
    if not v: raise ValueError(msg)


def save(p, v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as tmp:
        tmp.write((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())
    os.replace(tmp.name,p)


def journal(p,v):
    if p.exists(): check(read(p)==v, f"frozen receipt changed: {p}")
    else: save(p,v)


def gate():
    q=DEV/"held_verdict.json"
    if not q.exists():
        return "PENDING",None
    verdict=read(q)
    if verdict["decision"]!="PASS":
        return verdict["decision"],None
    p=read(PLAN)
    weights,package=DEV/"weights.json",read(DEV/"package.json")
    original=ROOT/"data/master3s2_source.py"
    check(p["formal"] is False and p["physical_seats"]==[0,1] and len(set(p["maps"]))==2
          and p["max_new_whole_games"]==8 and p["arms"]==["parent","candidate"],"wrong development schedule")
    check(verdict["checks"] and all(v is True for v in verdict["checks"].values()),"missing passing checks")
    check(read(DEV/"fit_verdict.json")["decision"]=="PASS","training gate not passed")
    check(sha(weights)==verdict["weights_sha256"]==package["weights_sha256"] and
          sha(DEV/"candidate.py")==verdict["package_sha256"]==package["candidate_sha256"] and
          sha(DEV/"native/held_labels.json")==verdict["label_sha256"],"held outcome binding differs")
    check(sha(DEV/"protocol.json")==package["protocol_sha256"]==read(weights)["protocol_sha256"]==p["rules_sha256"]
          and sha(HERE/"inventory.json")==p["inventory_sha256"],"protocol or source inventory changed")
    check(sha(PARENT)==p["parent_sha256"]==package["parent_sha256"] and sha(original)==p["original_sha256"],
          "parent or examiner changed")
    check(sha(subprocess.check_output(["git","show","HEAD:"+p["original_file"]],cwd=ROOT))==p["original_sha256"],
          "canonical original source differs")
    sys.path.insert(0,str(DEV))
    spec=importlib.util.spec_from_file_location("stratified_whole_build",DEV/"build.py")
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    serialized,reproduced=build.compose(read(weights))
    check(sha(serialized)==verdict["package_sha256"] and all(package[k]==v for k,v in reproduced.items()),
          "candidate cannot be reproduced from the frozen receiver's weights")
    binding={"held_verdict_sha256":sha(q),"protocol_sha256":p["rules_sha256"],"weights_sha256":sha(weights),
             "parent_sha256":sha(PARENT),"candidate_sha256":sha(serialized),"original_sha256":sha(original),
             "registration_sha256":sha(PLAN),"development_only":True}
    return "PASS",binding


def preflight():
    """Reissue the source-wide scan when needed; only identity checks, never a policy invocation."""
    plan=read(PLAN)
    for i,s in enumerate(plan["maps"]):
        target=HERE/f"pilot_preflight_{i}.zip"
        if target.exists():continue
        with tempfile.TemporaryDirectory(prefix="stratified-native-map-") as wd:
            out=Path(wd)/"audit.json"
            subprocess.run([sys.executable,str(SCANNER),"--seed",str(s),"--revision",plan["source_revision"],
                            "--declaration",str(PLAN.relative_to(ROOT)),"--registry",str(CATALOG),"--out",str(out)],
                           cwd=ROOT,env={**os.environ,"GIT_CONFIG_COUNT":"1","GIT_CONFIG_KEY_0":"grep.threads",
                                         "GIT_CONFIG_VALUE_0":"1"},check=True)
            item=zipfile.ZipInfo("identity.json");item.compress_type=zipfile.ZIP_DEFLATED
            temp=target.with_name(target.name+f".{os.getpid()}.tmp")
            with zipfile.ZipFile(temp,"w") as z:z.writestr(item,out.read_bytes())
            os.replace(temp,target)
    return clear()


def own_reference(ref, seed, board):
    """Allow a registration or a locally verifiable checked outcome. A seed/name alone is insufficient."""
    name=ref.get("path")
    digest=ref.get("sha256")
    if name==str(PLAN.relative_to(ROOT)):
        return digest==sha(PLAN) and seed in read(PLAN)["maps"]
    if not isinstance(name,str) or not isinstance(digest,str):
        return False
    src=(ROOT/name).resolve()
    if not src.is_file() or sha(src)!=digest or not src.is_relative_to(HERE):
        return False
    aggregate=src==HERE/"pilot_results.json"
    if not aggregate and not src.is_relative_to(OUTPUT):
        return False
    status,binding=gate()
    if status!="PASS" or not (OUTPUT/"binding.json").is_file() or read(OUTPUT/"binding.json")!=binding:
        return False
    from official_game_replay import check as replay
    candidates=list(OUTPUT.glob(f"*/receipt_{seed}_*.json"))
    for receipt in candidates:
        signed=read(receipt);game=signed["game"]
        if signed["binding"]!=binding or signed["official"]["map_sha256"]!=board:
            continue
        d=receipt.parent
        paths={receipt,d/game["raw_file"],d/f"row_{seed}_{game['candidate_seat']}.json"}
        if src not in paths and not aggregate:
            continue
        if aggregate:
            summary=read(src)
            if summary.get("binding")!=binding or summary.get("official_games")!=8 or summary.get("status")!="COMPLETE":
                continue
            if not any(x["seed"]==seed and x["seat"]==game["candidate_seat"] and x["arm"]==d.name for x in summary["same_map_both_seats"]):
                continue
        if signed["official"]==replay(d,game) and game["seed"]==seed:
            return True
    return False


def known_ownership(doc,boards):
    allowed=set()
    for s,b in boards.items():
        matches=[row for row in doc["maps"] if row["seed"]==s or row["initial_terrain_u8_sha256"]==b]
        if not matches:continue
        check(all((r["seed"],r["initial_terrain_u8_sha256"])==(s,b) for r in matches),
              "local catalog map aliases another registered seed")
        refs=[v for v in doc.get("sources",[]) if s in v.get("seeds",[]) or v.get("seed")==s]
        check(refs and all(own_reference(v,s,b) for v in refs),"local map has outside or unverifiable provenance")
        allowed.add(s)
    return allowed


def clear():
    sys.path.insert(0,str(BASE))
    import map_identity as mi
    from tree_win_gate import RESERVED_RANGE
    plan=read(PLAN)
    boards={}
    fp,ls,lb=mi.ladder_boards()
    _,ex_seeds,ex_boards=mi.exposed_boards([])
    planned={seed:mi.local_board(seed) for seed in plan["maps"]}
    known=known_ownership(read(BASE/mi.LOCAL_EXPOSURE),planned)
    for i,s in enumerate(plan["maps"]):
        p=HERE/f"pilot_preflight_{i}.zip"
        with zipfile.ZipFile(p) as archive: x=json.loads(archive.read("identity.json"))
        b=mi.local_board(s)
        check(x["candidate_seed"]==s and x["current_declaration"]["sha256"]==sha(PLAN)
              and x["candidate_initial_terrain_u8_sha256"]==x["official_opening_sha256"]==b
              and x["frozen_git_revision"]==plan["source_revision"] and x["sha256_of_this_script"]==sha(SCANNER)
              and x["historical_catalog_loaded_sha256"]==sha(CATALOG) and x["verdict"]=="eligible_in_checked_scope"
              and not x["same_terrain_matches"] and not x["conflicts"] and not x["reserved"]
              and not x["ladder_seed_or_terrain_match"],"historical world/seed clearance failed")
        check(s not in ls and b not in lb and (s in known or (s not in ex_seeds and b not in ex_boards))
              and not RESERVED_RANGE[0]<=s<RESERVED_RANGE[1],
              "current public or formal reservation overlap")
        boards[s]=b
    check(len(set(boards.values()))==2,"two physical maps required")
    return boards,{"source_sha256":sha(Path(mi.__file__)),
                   "ladder_digest":fp,"named_ladder_sources":len(mi.SOURCES),"distinct_ladder_seeds":len(ls),
                   "distinct_ladder_terrains":len(lb),"local_and_ladder_terrains":len(ex_boards)}


def newer_exposures(boards):
    """Check registered or running root checkouts for conflicts added after initial source-wide clearance."""
    audit=read(PLAN)
    regex=re.compile(rb"(?<![0-9])(?:" + b"|".join(str(s).encode() for s in boards) + rb")(?![0-9])|" +
                     b"|".join(b.encode() for b in boards.values()))
    # Reuse the repo scanner's interval grammar. Some reservations are recorded as ranges.
    spec=importlib.util.spec_from_file_location("whole_map_reservation_scanner",SCANNER)
    scanner=importlib.util.module_from_spec(spec);spec.loader.exec_module(scanner)
    roles=("protocol","registration","allocation","reservation","reserve","clearance","preflight","maps",
           "source_receipt","grant","scheduler","exposure","results","seed","raw_","plan","manifest","games")
    rawfiles=set()
    for root in ROOT.parent.iterdir():
        if not root.is_dir() or not (root/".git").exists():continue
        # Inspect changed paths for same-repository worktrees; covers streamed remote updates after the pinned cutoff.
        known=subprocess.run(["git","cat-file","-e",audit["source_revision"]+"^{commit}"],cwd=root,capture_output=True)
        if known.returncode:continue
        args=["git","diff","--name-only","-z",audit["source_revision"],"--","reports","data","automation"]
        result=subprocess.run(args,cwd=root,capture_output=True,check=True)
        other=subprocess.run(["git","ls-files","--others","--exclude-standard","-z","--","reports","data","automation"],
                             cwd=root,capture_output=True,check=True)
        for n in (result.stdout+other.stdout).split(b"\0"):
            if not n:continue
            rel=Path(os.fsdecode(n));p=root/rel
            if not p.is_file():continue
            if rel.is_relative_to(HERE.relative_to(ROOT)):continue
            if p.name.endswith((".json",".json.gz",".jsonl",".jsonl.gz",".md",".py",".tsv",".txt")):rawfiles.add(p)
    # Also detect freshly written plans in standalone job copies without Git metadata.
    found=subprocess.run(["find",str(ROOT.parent),"-type","f","-newer",str(PLAN),"(",
                          "-name","*.json","-o","-name","*.json.gz","-o","-name","*.jsonl.gz",
                          "-o","-name","*.md","-o","-name","*.py",")","-print0"],
                         capture_output=True,check=True)
    for n in found.stdout.split(b"\0"):
        if not n: continue
        p=Path(os.fsdecode(n))
        if p.is_relative_to(HERE) or ".git" in p.parts or not any(w in p.name.lower() for w in roles):continue
        rawfiles.add(p)
    for p in sorted(rawfiles):
        # Byte-identical copies of our own declared artifacts do not represent another experiment.
        data=p.read_bytes()
        if p.name==PLAN.name and sha(data)==sha(PLAN):continue
        if p.name=="exposure.json" and p.parent.name=="known_local_maps":
            known_ownership(json.loads(data),boards)
            continue
        encoded=gzip.decompress(data) if p.suffix==".gz" else data
        if regex.search(encoded):raise ValueError(f"seed/terrain mentioned in another current path: {p}")
        if not any(w in p.name.lower() for w in roles):continue
        if len(encoded)>4_000_000 or not any(str(s)[:2].encode() in encoded for s in boards):continue
        txt=encoded.decode("utf-8",errors="replace")
        for match in scanner.RANGE.finditer(txt):
            a,b=[int(v.replace("_","")) for v in match.groups()]
            if any(a<=s<=b for s in boards):raise ValueError(f"registered seed interval overlap: {p}")
        for match in scanner.PYRANGE.finditer(txt):
            a,b,step=match.groups();a,b=int(a.replace("_","")),int(b.replace("_",""));step=int(step.replace("_","")) if step else 1
            if any(s in range(a,b,step) for s in boards):raise ValueError(f"registered schedule overlap: {p}")
    return len(rawfiles)


def one(arm,seed,seat,board,binding,cpu):
    sys.path.insert(0,str(BASE))
    import tree_pack_arena
    import official_game_replay as official
    from tree_win_gate import validate_game,validate_examiner_configuration
    pack=PARENT if arm=="parent" else DEV/"candidate.py"
    target=OUTPUT/arm
    target.mkdir(parents=True,exist_ok=True)
    h=binding[arm+"_sha256"]
    wid=h if arm=="parent" else binding["weights_sha256"]
    saved=target/f"receipt_{seed}_{seat}.json"
    raw=target/f"raw_{seed}_{seat}.json.gz"
    workerrow=target/f"row_{seed}_{seat}.json"
    if saved.exists():
        previous=read(saved)
        check(previous["binding"]==binding and previous["official"]==official.check(target,previous["game"]),
              "saved result changed")
        return previous
    if not raw.exists():
        row=tree_pack_arena.arena.play((str(pack),wid,seed,seat,cpu,str(target),h))
        save(workerrow,row)
    t=read(raw)
    check(t["pack_sha256"]==h and t["weight_identity"]==wid and not t["errors"] and not t["audit_messages"]
          and t["examiner_md5"]=="57f9a594cce4f2ec9e3573d8d9d26ca1" and t["initial_terrain_u8_sha256"]==board
          and len(t["actions"])==t["steps"] and [x[0] for x in t["search_calls"]]==list(range(t["steps"]))
          and all(x[1] in {"searched","single","depth_zero_return_base"} for x in t["search_calls"])
          and any(x[1]=="searched" for x in t["search_calls"]), "raw exam or terrain not valid")
    validate_examiner_configuration(t["configuration"])
    if workerrow.exists():
        row=read(workerrow);validate_game(row)
    else:
        row={"seed":seed,"candidate_seat":seat,"actual_transitions":t["steps"],"raw_file":raw.name,
             "raw_sha256":sha(raw),"official_score_by_seat":t["official_score_by_seat"],"result":t["result"],
             "margin":t["margin"],"post_interruption":"complete action and per-call tag tape recovered; timing summaries unavailable"}
    result=official.check(target,row)
    check(result["map_sha256"]==board,"official map changed")
    receipt={"binding":binding,"game":row,"official":result}
    journal(saved,receipt)
    return receipt


def run(seconds):
    status,binding=gate()
    if status!="PASS":return {"status":status,"new_games":0}
    boards,coverage=clear()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    journal(OUTPUT/"binding.json",binding)
    end=time.monotonic()+seconds
    rows=[]
    for index,(seed,board) in enumerate(boards.items()):
        for seat in (0,1):
            for arm in (("parent","candidate") if (index+seat)%2==0 else ("candidate","parent")):
                ready=OUTPUT/arm/f"receipt_{seed}_{seat}.json"
                if not ready.exists() and time.monotonic()>=end:return {"status":"PAUSED_AT_GAME_BOUNDARY","complete":len(rows)}
                with LOCK.open("a") as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:return {"status":"SHARED_COURT_BUSY","complete":len(rows)}
                    st,now=gate();check(st=="PASS" and now==binding,"frozen gate or candidate changed")
                    cpu=min(os.sched_getaffinity(0))
                    if not ready.exists():
                        _,coverage=clear()
                        count=newer_exposures(boards)
                        save(OUTPUT/"latest_coordination.json",{"checked_current_paths":count,"ledger":coverage,"cpu":cpu,
                               "global_lease":str(LOCK.relative_to(ROOT)),"active_root_match_ports_allocated":0})
                    res=one(arm,seed,seat,board,binding,cpu)
                    rows.append({"arm":arm,"seed":seed,"seat":seat,"margin":res["game"]["margin"],"result":res["game"]["result"]})
    per_map=[]
    for seed in boards:
        pair=[r for r in rows if r["seed"]==seed]
        means={arm:sum(x["margin"] for x in pair if x["arm"]==arm)/2 for arm in ("parent","candidate")}
        per_map.append({"seed":seed,"mean_delta_margin":means["candidate"]-means["parent"],**means})
    final={"status":"COMPLETE","development_only":True,"official_games":8,"binding":binding,
           "same_map_both_seats":rows,"paired_map_margin":per_map,
           "mean_paired_margin":sum(x["mean_delta_margin"] for x in per_map)/2,
           "formal_games":0}
    check(len(rows)==8 and len({(r["seed"],r["seat"],r["arm"]) for r in rows})==8,
          "official development run incomplete")
    journal(HERE/"pilot_results.json",final)
    return final


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-if-passed",action="store_true")
    ap.add_argument("--preflight",action="store_true")
    ap.add_argument("--max-seconds",type=int,default=7200,help="stop scheduling games after this many seconds, finish any live game")
    args=ap.parse_args()
    if args.preflight:print(json.dumps(preflight(),sort_keys=True))
    elif args.run_if_passed:
        try:
            outcome=run(args.max_seconds)
        except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as err:
            outcome={"status":"STOPPED","reason":str(err),"formal_games":0}
        save(HERE/"pilot_status.json",outcome)
        print(json.dumps(outcome,sort_keys=True))
        if outcome["status"]=="COMPLETE":sys.exit(0)
        if outcome["status"] in {"PENDING","SHARED_COURT_BUSY","PAUSED_AT_GAME_BOUNDARY"}:sys.exit(75)
        sys.exit(2)
    else:print(json.dumps({"gate":gate()[0],"new_games":0}))

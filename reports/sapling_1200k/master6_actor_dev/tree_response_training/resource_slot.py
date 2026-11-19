"""Exclusive native-game handoff between sealed training intake 04 and the start of intake 05."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import gzip
import json
import os
from pathlib import Path
import re
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SLOT = HERE / "resources/veto_dev_slot"
DEV = HERE / "planter_survival/native_dev"
BEFORE = HERE / "batches/intake_04"
AFTER = HERE / "batches/intake_05"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


@contextmanager
def lock(name, *, blocking=True):
    with (SLOT / name).open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def seal(path, data):
    if path.exists():
        require(read(path) == data, f"existing receipt differs: {path}")
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def reservation():
    r = read(SLOT / "reservation.json")
    require(r["original_queue_sha256"] == sha(HERE / "next_intake.json")
            and r["source_protocol_sha256"] == sha(HERE / "protocol.json")
            and r["master_source_sha256"] == sha(ROOT / "data/master3s2_source.py")
            and r["dev_protocol_sha256"] == sha(DEV / "protocol.json")
            and r["dev"] == str(DEV.relative_to(HERE)), "training or examiner identity changed")
    return r


def evidence(items):
    require(isinstance(items, dict) and bool(items), "missing bound evidence files")
    out = {}
    for name, digest in items.items():
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts, "use a development- or repository-relative evidence path")
        matches = [p.resolve() for p in (DEV / rel, ROOT / rel) if p.is_file()]
        require(len(set(matches)) == 1, f"missing or ambiguous evidence path {name}")
        p = matches[0]
        require(p.is_relative_to(ROOT) and sha(p) == digest, f"evidence file changed: {name}")
        out[str(p.relative_to(ROOT))] = digest
    return out


def approval():
    r = reservation()
    doc = read(DEV / "slot_request.json")
    gate = read(DEV / "gate.json")
    require(gate.get("decision") == "run_native" and gate.get("started_native_games") == 0, "missing pre-game approval")
    gate_files = evidence(gate.get("evidence_files_sha256"))
    files = evidence(doc.get("evidence_files_sha256"))
    require(all(str((DEV / n).relative_to(ROOT)) in files for n in
                ("protocol.json", "release_manifest.json", "run.py", "candidate.py", "gate.json", "clearance.json")), "incomplete development request")
    protocol=read(DEV/"protocol.json")
    manifest=read(DEV/"release_manifest.json")
    require(protocol["maps"]==doc["seeds"] and protocol["physical_seats"]==r["physical_seats"]
            and protocol["cpu_affinity"]==r["cpus"] and protocol["examiner_sha256"]==r["master_source_sha256"]
            and manifest["protocol_sha256"]==r["dev_protocol_sha256"]
            and all(manifest[k]==protocol[k] for k in ("student_npz_sha256","risk_weights_sha256","parent_pack_sha256","parent_weight_manifest_sha256")),
            "different registered maps, resources or learning lineage")
    seeds = doc["seeds"]
    require(isinstance(seeds,list) and len(set(seeds)) == len(seeds) == 2 and all(type(s) is int for s in seeds), "two different maps required")
    from source_games import gate as examiner
    require(set(seeds).isdisjoint(read(HERE / "protocol.json")["development_maps"])
            and all(not examiner.RESERVED_RANGE[0] <= s < examiner.RESERVED_RANGE[1] for s in seeds),
            "training holdout or formal seed requested")
    forbidden = {k for b in read(HERE / "next_intake.json")["batches"] for k in b["maps"]}
    first = read(HERE / "clearance.json")["boards"]
    forbidden.update(map(int, first))
    forbidden_boards = set(first.values())
    for block in read(HERE / "next_intake.json")["batches"]:
        forbidden_boards.update(block["terrain_at_queue_time"].values())
    supplement=HERE/"supplement_01/registration.json"
    if supplement.exists():
        for block in read(supplement)["batches"]:
            forbidden.update(block["maps"])
            forbidden_boards.update(block["terrain_at_queue_time"].values())
    boards = {str(s): examiner.map_identity.local_board(s) for s in seeds}
    require(not (set(seeds) & forbidden) and len(set(boards.values())) == 2 and not (set(boards.values()) & forbidden_boards)
            and doc["boards"] == boards, "development map identity collision")
    cleared=read(DEV/"clearance.json")
    require(cleared["verdict"]=="eligible_in_checked_scope" and cleared["declaration_sha256"]==r["dev_protocol_sha256"]
            and cleared["boards"]==boards,"clearance does not certify the registered development boards")
    require(doc["examiner_md5"] == r["examiner_md5"] and set(doc["packs"]) == set(r["arms"]), "wrong examiner or policy arms")
    for pkg in doc["packs"].values():
        require(all(isinstance(pkg[k],str) and re.fullmatch(r"[a-f0-9]{64}",pkg[k]) for k in ("pack_sha256","weight_sha256")), "invalid pack identity")
    require(doc["packs"]["parent"] == {"pack_sha256": r["parent_pack_sha256"], "weight_sha256":r["parent_weight_sha256"]}
            and doc["packs"]["candidate"]["pack_sha256"] == sha(DEV / "candidate.py") == r["candidate_pack_sha256"] == manifest["pack_sha256"]
            and doc["packs"]["candidate"]["weight_sha256"] == sha(DEV/"release_manifest.json")
            and doc["packs"]["candidate"]["pack_sha256"] != r["parent_pack_sha256"], "incorrect parent or candidate pack")
    return {"slot_request_sha256":sha(DEV/"slot_request.json"), "prerequisite_files":gate_files,
            "development_files":files,"seeds":seeds,"boards":boards,"packs":doc["packs"]}


def training_started():
    d = AFTER / "source"
    return any(d.glob("raw_*")) or (d / "results.json").exists() or (d / ".source_schedule.lock").exists()


def native_started():
    return any((DEV / n).exists() and any((DEV / n).iterdir()) for n in ("parent", "candidate")) or (DEV / "comparison.json").exists()


def cancellation(reason):
    r = reservation();p = DEV / "gate.json";g = read(p)
    require(g.get("decision") in ("do_not_run_native","owner_withdrawn")
            and g.get("started_native_games") == 0, "only a frozen pre-play decision may cancel")
    if g["decision"] == "owner_withdrawn":
        require(g.get("owner") == r["owner"], "unknown withdrawing owner")
    files = evidence(g.get("evidence_files_sha256"))
    require(isinstance(reason,str) and len(reason.strip()) >= 4 and not native_started(), "reason missing or development began")
    return {"mode":"cancelled_before_play","reason":reason.strip(),"gate_sha256":sha(p),"evidence":files,
            "reservation_sha256":sha(SLOT/"reservation.json")}


def cancel(path, reason):
    require(path.resolve() == (DEV/"gate.json").resolve(), "cancellation must reference the registered gate")
    with lock(".native-owner.lock", blocking=False),lock(".state.lock"):
        require(not (SLOT/"release.json").exists(), "already released after completed games")
        seal(SLOT/"cancelled.json",cancellation(reason))


def validate_cancelled():
    existing=read(SLOT/"cancelled.json")
    require(existing == cancellation(existing.get("reason")), "invalid cancellation")
    return existing


def grant():
    from continue_intakes import verify_sealed, active_lock
    if not (BEFORE/"aggregate.json").is_file() or active_lock(BEFORE/".intake.schedule.lock"):return False
    if not (DEV/"slot_request.json").is_file():return False
    r = reservation()
    aggregate=read(BEFORE/"aggregate.json")
    require(aggregate["source_intakes"][-1] == {"dir":r["training_predecessor"],"labels_sha256":sha(BEFORE/"training_labels.json")}, "preceding intake not sealed")
    verify_sealed(BEFORE)
    signed={"boundary":r["pause_before"],"predecessor_aggregate_sha256":sha(BEFORE/"aggregate.json"),
            "reservation_sha256":sha(SLOT/"reservation.json"),**approval()}
    if not (SLOT/"granted.json").exists():
        from batch_cli import other_native
        if other_native(DEV):return False
        require(not training_started() and not native_started(), "native games already started without this reservation")
    seal(SLOT/"granted.json",signed)
    return True


def await_grant():
    while True:
        with lock(".state.lock"):
            if (SLOT/"cancelled.json").exists():
                validate_cancelled()
                raise RuntimeError("development was cancelled before play")
            if grant():return read(SLOT/"granted.json")
        time.sleep(5)


@contextmanager
def native_session():
    await_grant()
    with lock(".native-owner.lock", blocking=False):
        with lock(".state.lock"):
            require(not (SLOT/"cancelled.json").exists() and grant(), "native slot unavailable")
            require(not (SLOT/"release.json").exists(), "native block is already complete")
        yield


def completed(verify_official):
    signed = read(SLOT/"granted.json");r=reservation()
    require(not (SLOT/"cancelled.json").exists() and grant(), "grant differs or has been cancelled")
    from source_games import gate as examiner
    import official_game_replay as replayer
    expected={(s,p) for s in signed['seeds'] for p in r['physical_seats']}
    files={}
    outcomes={}
    for name in r["arms"]:
        d=DEV/name;report=read(d/"results.json");pkg=signed["packs"][name]
        head=report["header"];official=read(d/"official_replay.json")
        require(report["status"]=="complete" and head["seeds"]==signed["seeds"] and head["examiner_md5"]==r["examiner_md5"]
                and head["cpus"]==r["cpus"] and head["plan_sha256"] is None and all(head[k]==v for k,v in pkg.items()),"native identity or status differs")
        require(head["configuration"]==read(HERE/"source/results.json")["header"]["configuration"], "native environment configuration changed")
        rows=report["games"]
        require(len(rows)==4 and {(x["seed"], x["candidate_seat"]) for x in rows}==expected,"missing/repeated native games")
        replay=official["runs"][name]
        require(replay["pack_sha256"]==pkg["pack_sha256"] and len(replay["games"])==4, "official arm or count differs")
        by_key={(x["seed"],x["seat"]):x for x in replay["games"]}
        require(set(by_key)==expected and official["engine_source_sha256"]==read(HERE/"source/official_replay.json")["engine_source_sha256"], "incomplete official coverage or wrong engine")
        for row in rows:
            s,p=row["seed"],row["candidate_seat"];rr=by_key[s,p]
            require(row["raw_file"]==f"raw_{s}_{p}.json.gz" and rr["map_sha256"]==signed["boards"][str(s)]
                    and rr["transcript_sha256"]==row["raw_sha256"]==sha(d/row["raw_file"])
                    and rr["statuses"]==["DONE","DONE"] and rr["steps"]==row["actual_transitions"]
                    and rr["official_reward_by_seat"]==row["official_score_by_seat"] and rr["result"]==row["result"], "unbound official replay")
            raw=json.loads(gzip.decompress((d/row["raw_file"]).read_bytes()))
            examiner.validate_examiner_configuration(raw["configuration"])
            require(not raw["audit_messages"], "examiner observer reported an error")
            examiner.validate_game(row)
            if verify_official:
                examiner.verify_raw(d,row,pkg["pack_sha256"],pkg["weight_sha256"],expected_terrain=signed["boards"][str(s)])
                require(replayer.check(d,row)==rr,"independent official interpreter disagrees")
            outcomes[name,s,p]=rr
            files[f"{name}/{row['raw_file']}"]=row["raw_sha256"]
        for f in ("results.json","official_replay.json"):files[f"{name}/{f}"]=sha(d/f)
    comparison=read(DEV/"comparison.json")
    pairs=comparison.get("pairs",[])
    require(comparison.get("games")==8 and comparison.get("games_per_arm")==4 and len(pairs)==4
            and {(p["seed"],p["seat"]) for p in pairs}==expected and all(all(arm in p for arm in r["arms"]) for p in pairs),
            "comparison does not cover the same eight native games")
    require(comparison.get("grant_sha256")==sha(SLOT/"granted.json")
            and comparison.get("gate_sha256")==sha(DEV/"gate.json")
            and comparison.get("release_manifest_sha256")==sha(DEV/"release_manifest.json"), "comparison identity differs")
    for pair in pairs:
        for name in r["arms"]:
            rr=outcomes[name,pair["seed"],pair["seat"]]
            require(all(pair[name].get(k)==v for k,v in rr.items()), "comparison contains a different official outcome")
    files["comparison.json"]=sha(DEV/"comparison.json")
    return {"mode":"completed_8_official_games","games":8,"granted_sha256":sha(SLOT/"granted.json"),"verified_files":files}


def release():
    with lock(".native-owner.lock", blocking=False),lock(".state.lock"):
        receipt=completed(verify_official=True)
        seal(SLOT/"release.json",receipt)
        return receipt


def wait_before_source(wd, plan):
    if Path(wd).resolve()!=AFTER:return
    require(plan.get("intake_name")==AFTER.name,"unexpected training block")
    while True:
        with lock(".state.lock"):
            if (SLOT/"cancelled.json").is_file():
                validate_cancelled();return
            if (SLOT/"release.json").is_file():
                require(read(SLOT/"release.json")==completed(verify_official=False),"invalid development release")
                return
            grant()
        time.sleep(5)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["await-grant","release","cancel"])
    parser.add_argument("--reason",default="")
    parser.add_argument("--evidence",type=Path)
    args=parser.parse_args()
    if args.action=="await-grant":print(json.dumps(await_grant()))
    elif args.action=="release":print(json.dumps(release()))
    else:
        if not args.evidence:parser.error("cancel requires --evidence and --reason")
        cancel(args.evidence,args.reason)

"""Count a specific empty-worker trade throughout the frozen 1,528 scored games, without selecting winners."""
import gzip
import io
import tarfile
from functools import lru_cache
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TOP = HERE.parents[1] / "all_top5"
sys.path.insert(0, str(TOP))
import fetch
from continue_appendix import previous_manifests
from defense import DEST, MOVE
from scan import summary

LATEST = TOP / "appendix_aca9b8a3"
OUT = HERE / "frequency_games.jsonl.gz"


@lru_cache(maxsize=1)
def archive_bytes(relative):
    path = fetch.ROOT / relative
    return path.read_bytes() if path.exists() else fetch.git("show", "HEAD:" + relative)


def packed_replay(game):
    path = fetch.CACHE / game["path"]
    if path.exists():
        return path.read_bytes()
    location = game["archive_location"]
    with tarfile.open(fileobj=io.BytesIO(archive_bytes(location["archive"]))) as archive:
        return archive.extractfile(location["member"]).read()


def extract(game, raw):
    doc = json.loads(gzip.decompress(raw))
    frames = doc["steps"]
    config = doc["configuration"]
    T = len(frames)-1
    if not game["frame_verified"]:
        raise ValueError("unverified replay")
    if T != game["transitions"] or doc["rewards"] != game["score_by_seat"]:
        raise ValueError(game["id"])
    observed = [frame[0]["observation"]["players"] for frame in frames]
    # Histories per real seat speed up follow-up checks and preserve the exact observed recovery time.
    collectors = [[{uid:w for uid,w in p[seat][2].items() if w[2] == "COLLECTOR"} for p in observed] for seat in (0,1)]
    roots = [Counter(),Counter()]
    cases = []
    registered = [seat for seat in (0,1) if game["seat_strategy_ids"][seat]]
    for t in range(T):
        pre = observed[t]
        for hunter_seat in registered:
            counts = roots[hunter_seat]
            counts["all"] += 1
            target = 1-hunter_seat
            price = config["recCollectorCost"] + config["recCollectorIncreaseCost"] * (len(collectors[0][t]) + len(collectors[1][t]))
            if pre[target][0] >= price or len(collectors[target][t]) != 1:
                continue
            counts["low_one_collector"] += 1
            victim, (vcell, cargo, _) = next(iter(collectors[target][t].items()))
            if cargo != 0:
                continue
            counts["low_one_empty_collector"] += 1
            bases = set(pre[0][1].values()) | set(pre[1][1].values())
            possible = set(DEST[vcell])-bases
            hunters = {wid:pos for wid,(pos,load,_) in collectors[hunter_seat][t].items()
                       if load == 0 and possible.intersection(DEST[pos])}
            if not hunters:
                continue
            counts["trade_geometry"] += 1
            actions = [r["action"] or {} for r in frames[t+1]]
            v_end = DEST[vcell][MOVE[actions[target].get(victim)]]
            hunters_end = {wid:DEST[pos][MOVE[actions[hunter_seat].get(wid)]] for wid,pos in hunters.items()}
            if any(cell in possible for cell in hunters_end.values()):
                counts["selected_possible_cell"] += 1
            met = [wid for wid,cell in hunters_end.items() if cell == v_end]
            follow = collectors[target][t+1:min(T+1,t+31)]
            if t+30 <= T:
                counts["geometry_full30"] += 1
                if not any(follow):
                    counts["geometry_no_collector_full30"] += 1
            if not met or v_end not in possible:
                continue
            counts["same_cell"] += 1
            landers = [(s,wid) for s in (0,1) for wid,(pos,_,_) in pre[s][2].items()
                       if DEST[pos][MOVE[actions[s].get(wid)]] == v_end]
            if len(landers) != 2 or len(met) != 1:
                continue
            hunter = met[0]
            if hunter in observed[t+1][hunter_seat][2] or victim in observed[t+1][target][2]:
                raise ValueError("two equal empty collectors met on open land but did not both disappear")
            counts["unambiguous_trade"] += 1
            regained = next((k for k in range(t+1,T+1) if collectors[target][k]),None)
            if not collectors[target][t+1]:
                counts["trade_zero_collectors_after"] += 1
            if t+30 <= T:
                counts["trade_full30"] += 1
                if not any(follow):
                    counts["trade_no_collector_full30"] += 1
            if regained is None:
                counts["trade_never_rehired"] += 1
            cases.append({"pre_t":t,"attack_seat":hunter_seat,"hunter":hunter,"victim":victim,
                          "before_cells":[hunters[hunter],vcell],
                          "commands":[actions[hunter_seat].get(hunter) or "NONE",actions[target].get(victim) or "NONE"],
                          "cell":v_end,"target_before":summary(pre,target,config),
                          "target_after":summary(observed[t+1],target,config),
                          "first_real_collector_after_t":regained,
                          "last_observed_state":summary(observed[-1],target,config)})
    return {"id":game["id"],"map_seed":game["seed"],"replay_sha256":game["sha256"],
            "names":game["seat_names"],"registered":registered,"scores":game["score_by_seat"],"steps":T,
            "counts":roots,"cases":cases}


def main():
    prior = previous_manifests(LATEST)
    manifests = [fetch.load(p) for p in prior]
    cohort = [g for m in manifests for g in m["games"]]
    distinct = len({g["id"] for g in cohort})
    if len(cohort) != distinct or distinct != manifests[-1]["accumulated_games"] or distinct != 1528:
        raise ValueError("entire frozen cohort required")
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    traces = Path("/tmp") / ("sapling-ladder-fleet-" + digest)
    traces.mkdir(exist_ok=True)
    names = set(fetch.NAMES)
    data = []
    for i,g in enumerate(cohort):
        target = traces / (g["sha256"]+".json")
        if target.is_file():
            row = fetch.load(target)
            if row["replay_sha256"] != g["sha256"] or row["id"] != g["id"]:
                raise ValueError("cached receipt identity mismatch")
        else:
            raw = packed_replay(g)
            if fetch.sha(raw) != g["sha256"]:
                raise ValueError("wrong raw replay")
            row = extract(g,raw)
            fetch.dump(target,row)
        data.append(row)
        if not (i+1)%200:
            print(i+1,flush=True)
    strata = {}
    for r in data:
        for s in r["registered"]:
            n = r["names"][s]
            if n not in names:
                raise ValueError("unregistered appearance")
            res = r["scores"][s]-r["scores"][1-s]
            result = "W" if res>0 else ("L" if res<0 else "D")
            for k in (n, "all_appearances", (n+"|seat"+str(s)), (n+"|"+result)):
                entry=strata.setdefault(k,{"counts":Counter(),"appearing_in_games":Counter()})
                entry["counts"].update(r["counts"][s]);entry["counts"]["appearances"]+=1
                entry["appearing_in_games"].update([c for c,v in r["counts"][s].items() if v>0])
    fetch.atomic(OUT,gzip.compress(b"".join((json.dumps(r,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode() for r in data),mtime=0))
    item={"revision":manifests[-1]["revision"],"normal_games":len(cohort),"excluded_error_games":1,
          "numeric_joint_steps":sum(r["steps"] for r in data),"ledger_sha256":fetch.sha(OUT.read_bytes()),"strata":strata}
    fetch.dump(HERE/"frequency_summary.json",item)
    print("finished",item["normal_games"],flush=True)


if __name__=="__main__":
    main()

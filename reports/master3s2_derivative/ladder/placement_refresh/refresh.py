"""Read all public cards of this strategy and validate this placement supplement."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import ssl
import statistics
import sys
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
PREVIOUS = LADDER / "v1_placement_review"
ROOT = HERE.parents[3]
ID = "b99742a4a510a4de5dc99639b3c85604"
BASE = "https://117.72.210.134"
sys.path[:0] = [str(PREVIOUS), str(ROOT / "reports/master_family/deep_review")]
import review
import compare
from recover_public import extract, canonical, validate


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def store(path, value):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def get(url):
    request = Request(url, headers={"User-Agent": "Carbon League public replay reader", "Accept-Encoding": "gzip"})
    with urlopen(request, context=ssl._create_unverified_context(), timeout=40) as reply:
        data = reply.read()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def fetch(path):
    pages = []
    offset = 0
    while True:
        url = f"{BASE}/api/matches?limit=20&offset={offset}&strategy={ID}"
        data = json.loads(get(url))
        pages.append({"url": url, "fetched_utc": datetime.now(timezone.utc).isoformat(), "data": data})
        following = data["next_offset"]
        if following is None:
            break
        assert following > offset
        offset = following
    ids = [m["id"] for page in pages for m in page["data"]["matches"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate IDs between live pages; repeat the GET snapshot")
    store(path, pages)


def score(m, g):
    own = "a" if m["a"] == ID else "b"
    opp = "b" if own == "a" else "a"
    assert m[own] == ID and m[opp] != ID
    margin = Decimal(str(g["scores"][own])) - Decimal(str(g["scores"][opp]))
    outcome = {own: "W", opp: "L", "draw": "D"}[g["winner"]]
    return dict(match=m["id"], game=g["game"], created=m["created"], finished=m["finished"],
                kind=m["kind"], rated=m["rated"], opponent=m["name_"+opp], opponent_id=m[opp],
                seat=g["order"].index(own), score=[g["scores"][own], g["scores"][opp]], margin=float(margin),
                outcome=outcome, statuses=g["statuses"], steps=g["steps"], wall=g["duration_seconds"],
                official_map=g["map_sha256"], ordered_names=[m["name_"+s] for s in g["order"]])


def main(path, online):
    pages = read(path)
    prior = read(PREVIOUS / "ledger.json.gz")
    old = {(g["match_id"], g["game"]): g for g in prior["registered_games"]}
    all_matches = [m for p in pages for m in p["data"]["matches"]]
    assert len({m["id"] for m in all_matches}) == len(all_matches)
    games, pending, placement, new, controls, real_terrain = [], [], [], [], [], {}
    loss_ids = {c["opponent_id"] for c in old.values() if c["outcome"] == "L"}
    for m in sorted(all_matches,key=lambda m:(m["created"], m["id"])):
        assert ID in (m["a"], m["b"])
        if m["state"] != "done" or not m["result"]:
            pending.append({k: m.get(k) for k in ["id","a","b","name_a","name_b","kind","rated","state","game_count","created","finished"]})
            continue
        assert len(m["result"]) == m["game_count"]
        for g in m["result"]:
            c = score(m, g)
            games.append(c)
            is_placement = c["kind"] == "newrank-placement" and bool(c["rated"])
            is_control = c["opponent_id"] in loss_ids and c["kind"] == "newrank-ladder" and bool(c["rated"])
            if not (is_placement or is_control):
                continue
            key = c["match"], c["game"]
            if is_placement:
                placement.append(c)
            replay_name = f"{key[0]}-{key[1]}.json.gz"
            if key in old:
                original = old[key]
                assert [original[k] for k in ("opponent","opponent_id","score","margin","outcome","seat","steps")] == [c[k] for k in ("opponent","opponent_id","score","margin","outcome","seat","steps")]
                archive = LADDER/"replays"/replay_name
                assert hashlib.sha256(archive.read_bytes()).hexdigest() == original["replay_sha256"]
                real_terrain[key] = original["terrain_sha256"]
                continue
            archive = HERE/"replays"/replay_name
            if not archive.exists() and online:
                env, _, _ = extract(get(f"{BASE}/replay/{key[0]}/{key[1]}"))
                validate(env, {**m, **g})
                archive.parent.mkdir(exist_ok=True)
                archive.write_bytes(gzip.compress(canonical(env), compresslevel=6, mtime=0))
            if not archive.exists():
                raise ValueError(f"download public replay with --fetch-new-replays: {replay_name}")
            env = read(archive)
            c["replay_verified"] = validate(env, {**m, **g})
            c["replay_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            c["opening_carbon_hash"] = hashlib.sha256(bytes(env["steps"][0][0]["observation"]["full_carbon"])).hexdigest()
            if is_placement:
                real_terrain[key] = c["opening_carbon_hash"]
            if (c["outcome"] == "L" or is_control) and c["steps"] == 299 and set(c["statuses"].values()) == {"DONE"}:
                # The existing economics module resolves its input as HERE.parent / 'replays'.
                compare.HERE = HERE/"analysis"
                details = review.losing_game(dict(opponent=c["opponent"], match_id=key[0], seat=c["seat"], steps=c["steps"]), archive)
                details["card"] = c
                store(HERE/f"{key[0]}-{key[1]}.ledger.json.gz", details)
            (new if is_placement else controls).append(c)
    assert set(old) <= set(real_terrain)
    assert len({(c["match"],c["game"]) for c in games}) == len(games)
    opponents = defaultdict(list)
    names = defaultdict(set)
    for c in placement:
        opponents[c["opponent_id"]].append(c)
        names[c["opponent"]].add(c["opponent_id"])
    abnormal = [c for c in placement if len(set(c["statuses"].values())) != 1 or list(c["statuses"].values())[0] != "DONE"]
    summary = dict(snapshot_utc=pages[-1]["fetched_utc"], public_strategy=ID, raw_pages_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   cards=len(all_matches), by_kind=dict(Counter(m["kind"] for m in all_matches)), multi_game_cards=[m["id"] for m in all_matches if m["game_count"]>1],
                   pending=pending, all_settled_games=games, placement=len(placement), unchanged_old=len(old), new_placement=new, same_opponent_nonplacement_controls=controls,
                   outcomes=dict(Counter(c["outcome"] for c in placement)), seat_outcomes={s:dict(Counter(c["outcome"] for c in placement if c["seat"]==s)) for s in (0,1)},
                   mean_margin=statistics.mean(c["margin"] for c in placement), median_margin=statistics.median(c["margin"] for c in placement),
                   first_placement_created_utc=datetime.fromtimestamp(min(c["created"] for c in placement),timezone.utc).isoformat(),
                   last_placement_finished_utc=datetime.fromtimestamp(max(c["finished"] for c in placement),timezone.utc).isoformat(),
                   unique_opponents=len(opponents), duplicate_names_across_ids={k:sorted(v) for k,v in names.items() if len(v)>1},
                   repeat_opponents=[[dict(opponent=c["opponent"], id=i, match=c["match"],margin=c["margin"],seat=c["seat"]) for c in values] for i,values in opponents.items() if len(values)>1],
                   unique_maps=len(set(real_terrain.values())), abnormal_placement=abnormal, placement_steps=dict(Counter(c["steps"] for c in placement)),
                   losses=[c for c in placement if c["outcome"]=="L"])
    store(HERE / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-cards",type=Path,help="public GET complete strategy-card pages into given fresh file")
    parser.add_argument("--cards",type=Path,default=HERE/"final_readback.json.gz")
    parser.add_argument("--fetch-new-replays",action="store_true")
    args = parser.parse_args()
    if args.fetch_cards:
        fetch(args.fetch_cards)
    else:
        main(args.cards, args.fetch_new_replays)

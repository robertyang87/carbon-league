"""Fetch public match and losing replay data for precisely the registered ds-pro-v2 strategy."""
import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
import sys

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
sys.path.insert(0, str(LADDER / "placement_refresh"))
import refresh

ID = "ec94599e5182be7d35eb14bc288625cd"
refresh.ID = ID


def audit(snapshot, online):
    pages = refresh.read(snapshot)
    matches = [m for p in pages for m in p["data"]["matches"]]
    by_match = {m["id"]: m for m in matches}
    assert len(by_match) == len(matches) and pages[-1]["data"]["next_offset"] is None
    games, pending, losses, wins_same_opponent = [], [], [], []
    losing_ids = {c["opponent_id"] for m in matches if m["state"]=="done" for g in (m["result"] or [])
                  if (c := refresh.score(m,g))["outcome"]=="L"}
    for match in sorted(matches, key=lambda m: (m["created"], m["id"])):
        assert ID in (match["a"], match["b"])
        if match["state"] != "done" or not match["result"]:
            pending.append({key:match.get(key) for key in ("id","a","b","name_a","name_b","kind","created","finished","state","rated","game_count")})
            continue
        assert len(match["result"]) == match["game_count"]
        for game in match["result"]:
            card = refresh.score(match, game)
            games.append(card)
            if card["outcome"] != "L" and not (card["outcome"] == "W" and card["opponent_id"] in losing_ids):
                continue
            key = card["match"] + "-" + str(card["game"])
            archive = HERE / "replays" / (key + ".json.gz")
            if not archive.exists():
                old = LADDER / "replays" / archive.name
                if old.exists():
                    archive.write_bytes(old.read_bytes())
                elif online:
                    from recover_public import extract, canonical
                    replay, _, _ = extract(refresh.get(f"{refresh.BASE}/replay/{card['match']}/{card['game']}"))
                    refresh.validate(replay, {**match, **game})
                    archive.write_bytes(gzip.compress(canonical(replay), compresslevel=6, mtime=0))
                else:
                    raise ValueError(f"Missing {archive.name}, run with --download")
            env = refresh.read(archive)
            check = refresh.validate(env,{**match,**game})
            assert check["scores_checked"] and check["both_seats_checked"] and check["steps_checked"]
            card["verified"] = check
            card["raw_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            card["opening_carbon_hash"] = hashlib.sha256(bytes(env["steps"][0][0]["observation"]["full_carbon"])).hexdigest()
            print("archived",card["opponent"],key,card["score"],card["statuses"],flush=True)
            detail_path = HERE / (key + ".ledger.json.gz")
            if card["steps"] == 299 and set(card["statuses"].values()) == {"DONE"}:
                refresh.compare.HERE = HERE/"analysis"
                detail = refresh.review.losing_game(dict(opponent=card["opponent"],match_id=card["match"],seat=card["seat"],steps=card["steps"]),archive)
            else:
                proof = refresh.review.validation.official({"path":archive.name},archive.parent)
                assert proof["matched"] == card["steps"] and proof["first_divergence"] is None
                detail = {"official":proof,"steps_verified":card["steps"],"attribution":"technical or shortened result; see official statuses and verbatim replay","joint_commands":[[s["action"] or {} for s in pair] for pair in env["steps"][1:]]}
            detail["card"] = card
            refresh.store(detail_path,detail)
            (losses if card["outcome"] == "L" else wins_same_opponent).append(card)
    assert len({(c["match"],c["game"]) for c in games}) == len(games)
    groups = {}
    for kind in sorted({m["kind"] for m in matches}):
        group = [g for g in games if g["kind"]==kind]
        groups[kind] = dict(cards=sum(m["kind"]==kind for m in matches),completed=len(group),
            rated=sum(bool(c["rated"]) for c in group),outcomes=dict(Counter(c["outcome"] for c in group)),
            margins_mean=mean(c["margin"] for c in group) if group else None,
            margins_median=median(c["margin"] for c in group) if group else None,
            seats={s:dict(Counter(c["outcome"] for c in group if c["seat"]==s)) for s in (0,1)},
            opponents=len({c["opponent_id"] for c in group}),
            different_scorecard_maps=len({c["official_map"] for c in group}),
            scheduled_games=sum(m["game_count"] for m in matches if m["kind"]==kind),
            pending=sum(m["kind"]==kind for m in pending))
    start = min(m["created"] for m in matches)
    end = max(c["finished"] for c in games)
    summary = dict(utc=pages[-1]["fetched_utc"], snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        strategy=ID, observed_package=refresh.read(LADDER/"ds-pro-v2.json")["candidate_sha256"], groups=groups,
        first_created=datetime.fromtimestamp(start,timezone.utc).isoformat(), last_done=datetime.fromtimestamp(end,timezone.utc).isoformat(),
        pending=pending, matches=len(matches),games=games,losses=losses,wins_over_losing_opponents=wins_same_opponent)
    refresh.store(HERE/"summary.json",summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fetch",type=Path)
    parser.add_argument("--pages",type=Path,default=HERE/"final.json.gz")
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    if args.fetch: refresh.fetch(args.fetch)
    else: audit(args.pages,args.download)

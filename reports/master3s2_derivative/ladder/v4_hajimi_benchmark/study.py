"""Reconcile six author-bound ladder meetings across three DS versions and trace the latest loss."""

from pathlib import Path
from collections import Counter
import gzip
import json
import hashlib
import sys

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
sys.path.insert(0, str(LADDER / "placement_refresh"))
import refresh

OPP = "66ab03a1462c6fc6ce366a83dca18827"
IDS = {
    "v4": "16a3cd313829ffe78bd24ac9714b2400",
    "v2": "ec94599e5182be7d35eb14bc288625cd",
    "v1": "b99742a4a510a4de5dc99639b3c85604",
}
FOCUS = "687ecf4fee697c81406fe6bbc4735141"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def file_info(p):
    return {"path": str(p.relative_to(LADDER)), "sha256": sha(p.read_bytes())}


def choose_file(m, match):
    key = m["id"] + "-1"
    dest = HERE / "replays" / (key + ".json.gz")
    if dest.exists():
        return dest
    old = sorted(
        p
        for p in LADDER.rglob(key + ".json.gz")
        if p.parent.name == "replays" and not p.is_relative_to(HERE)
    )
    if old:
        return old[0]
    env, _, _ = refresh.extract(refresh.get(refresh.BASE + "/replay/" + m["id"] + "/1"))
    refresh.validate(env, m | match)
    dest.write_bytes(gzip.compress(refresh.canonical(env), compresslevel=6, mtime=0))
    return dest


def facts(raw):
    m = raw["match"]
    version = raw["version"]
    refresh.ID = IDS[version]
    g = m["result"][0]
    assert OPP in (m["a"], m["b"]) and m["state"] == "done" and len(m["result"]) == 1
    c = refresh.score(m, g)
    assert c["opponent"] == "哈基米基准" and c["opponent_id"] == OPP
    path = choose_file(m, g)
    game = refresh.read(path)
    refresh.validate(game, m | g)
    key = m["id"] + "-1.ledger.json.gz"
    old = sorted(p for p in LADDER.rglob("*" + key) if not p.is_relative_to(HERE))
    lp = (HERE / key) if (HERE / key).exists() or not old else old[0]
    if not lp.exists():
        refresh.compare.HERE = path.parent.parent / "analysis"
        l = refresh.review.losing_game(
            dict(
                opponent=c["opponent"],
                match_id=c["match"],
                seat=c["seat"],
                steps=c["steps"],
            ),
            path,
        )
        refresh.store(lp, l)
    else:
        l = refresh.read(lp)
    assert (
        l["official"]["matched"] == l["fast_verified"] == 299
        and l["official_scores"] == c["score"]
        and l["replay_sha256"] == sha(path.read_bytes())
    )
    frames = game["steps"]
    s = c["seat"]
    recruits = [Counter(), Counter()]
    disappeared = [Counter(), Counter()]

    def walk(who, uid, until):
        rows = []
        for t in sorted({max(until - k, 0) for k in (10, 5, 3, 2, 1, 0)}):
            o = frames[t][0]["observation"]
            w = o["players"][who][2].get(uid)
            rows.append(
                dict(
                    t=t,
                    worker=w,
                    command=(frames[t + 1][who]["action"] or {}).get(uid, "NONE")
                    if w
                    else None,
                )
            )
        return rows

    def window(a, b):
        flows = []
        for t in range(a, b):
            for e in l["receipts_by_transition"][t]:
                flows.append(
                    dict(**e, arrive=t + 1, approach=walk(e["seat"], e["worker"], t))
                )
        return {
            "from": a,
            "to": b,
            "margin_before": l["steps"][a]["margin"],
            "margin_after": l["steps"][b]["margin"],
            "bank": flows,
        }

    for t in range(299):
        before = frames[t][0]["observation"]["players"]
        after = frames[t + 1][0]["observation"]["players"]
        for i in (0, 1):
            for uid in after[i][2].keys() - before[i][2].keys():
                recruits[i][after[i][2][uid][2]] += 1
            for uid in before[i][2].keys() - after[i][2].keys():
                disappeared[i][before[i][2][uid][2]] += 1
    latest = max(
        [r for r in l["all_streaks"] if r["sign"] == -1],
        key=lambda a: a["end"],
        default=None,
    )
    longest = l["longest_deficit"]
    targets = {v["end"] for v in l["worst_windows"].values()} | {longest["start"]}
    if latest:
        targets.add(latest["start"])
    result = dict(
        card=c,
        replay=file_info(path),
        ledger=file_info(lp),
        terrain=sha(bytes(frames[0][0]["observation"]["full_carbon"])),
        long=longest,
        latest=latest,
        counts=l["sign_counts"],
        last_lead=l["last_strict_lead"],
        worst=l["worst"],
        whole=l["whole"],
        phases=l["phases"],
        planting=[
            {k: v for k, v in z.items() if k != "individual"} for z in l["planting"]
        ],
        initial=[(frames[1][i]["action"] or {}) for i in (s, 1 - s)],
        births=[recruits[i] for i in (s, 1 - s)],
        deaths=[disappeared[i] for i in (s, 1 - s)],
        samples={
            t: l["steps"][t]
            for t in (0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299)
        },
    )
    if m["id"] == FOCUS:
        result.update(
            windows={
                f"{t}_{w}": window(max(0, t - w), t)
                for t in sorted(targets)
                for w in (3, 5, 10)
            }
        )
    return result


def main():
    out = []
    for c in refresh.read(HERE / "scorecards.json"):
        q = facts(c)
        out.append(q)
        print(
            q["card"]["match"],
            q["card"]["margin"],
            q["counts"],
            q["long"],
            q["whole"]["difference"],
            flush=True,
        )
    assert (
        len(out) == len({r["terrain"] for r in out}) == 6
        and len({r["card"]["opponent_id"] for r in out}) == 1
    )
    refresh.store(HERE / "comparison.json", out)


if __name__ == "__main__":
    main()

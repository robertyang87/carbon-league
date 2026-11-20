"""Sign two complete physical orders selected from each recorded training map without its response."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PREVIOUS = STUDY / "midgame_economic_policy/native_response"
sys.path.insert(0, str(PREVIOUS))
import select_roots

common = select_roots.common
need = common.need
HOSTED = STUDY / "policy_window_returns/hosted"
MANIFEST = HOSTED / "manifest.json"
PACKAGE = STUDY / "native_action_policy/candidate.py"
PROPOSER = "e830229d23817cdcb8ad7e7e392cb3ae57cb1e04503e70ecefe91bf446fc20d4"
TRAIN_SEEDS = (621758540, 621758541, 621758542, 621758543)
TICKS = (258, 264, 270, 276, 282)


def sha(path):
    return common.sha(path.read_bytes())


def read_json(path):
    return json.loads(path.read_bytes())


def source_for(seed):
    manifest = read_json(MANIFEST)
    tickets = [t for t in manifest["tickets"] if t["seed"] == seed and t["role"] == "fit"]
    need(len(tickets) == 1 and seed in TRAIN_SEEDS, "Missing or extraneous source map")
    t = tickets[0]
    archive = HOSTED / (str(t["artifact"]) + ".zip")
    need(sha(archive) == t["sha256"], "Wrong archive")
    with zipfile.ZipFile(archive) as z:
        signed = z.read("verified.json")
        raw = z.read("source.json.gz")
        need(common.sha(signed) == t["verification_receipt_sha256"], "Wrong verification receipt")
        raw_sha = common.sha(raw)
        need(raw_sha == json.loads(signed)["source_sha256"], "Wrong source trajectory")
    source = json.loads(gzip.decompress(raw))
    need(source["actors"]["bc"] == common.SIGNED["bc"] and
         source["actors"]["original"] == common.SIGNED["original"] and
         source["seed"] == seed and source["seat"] == t["seat"] and source["kind"] == "fit",
         "Mismatch in actor, seat or use")
    return source, {"seed": seed, "seat": t["seat"], "archive_sha256": t["sha256"],
                    "source_sha256": raw_sha, "terrain": source["terrain"]}


def pick(views, cfg):
    candidate = common.collect.Player(PACKAGE)
    parent = common.collect.Player(common.BC)
    scanned = []
    for obs in views:
        t = obs["step"]
        if t == 0:
            for player in (candidate, parent):
                player.act(obs, cfg)
            continue
        if t not in TICKS:
            for player in (candidate, parent):
                common.observe_only(player, obs)
            continue
        sent = candidate.act(obs, cfg)
        info = candidate.mod._LAST_POLICY
        base = parent.act(obs, cfg)
        options = info["orders"]
        need(info["step"] == t and info["seat"] == obs["player"] and info["parent"] == base and
             any(sent == row["action"] for row in options), "Action identities disagree")
        need(len({common.digest(r["action"]) for r in options}) == len(options), "Duplicate order")
        for key in ("probability", "prior_probability"):
            need(all(r[key] > 0 for r in options) and
                 math.isclose(sum(r[key] for r in options), 1., abs_tol=1e-8), "Bad distribution")
        base_info = next(r for r in options if r["action"] == base)
        changed = [r for r in options if r["action"] != base]
        scanned.append({"t": t, "support": len(options), "public": common.digest(obs)})
        if not changed:
            continue
        best = min(changed, key=lambda r: (-r["probability"], common.digest(r["action"])))
        change = (math.log(best["probability"] / base_info["probability"])
                  - math.log(best["prior_probability"] / base_info["prior_probability"]))
        scanned[-1]["learned_logodds_delta"] = change
        if abs(change) > 1e-5:
            return {"selected": True, "root_step": t, "root_public_sha256": common.digest(obs),
                    "orders": {"parent": base, "alternative": best["action"]},
                    "pivot_role": info["pivot_role"], "pivot_uid": info["pivot_uid"],
                    "menu": options, "logodds_delta": change, "scan": scanned}
    return {"selected": False, "scan": scanned}


def main():
    need(sha(PACKAGE) == PROPOSER and sha(common.BC) == common.SIGNED["bc"], "Package drift")
    tickets = []
    for seed in TRAIN_SEEDS:
        source, ident = source_for(seed)
        cfg = {k: v for k, v in source["configuration"].items() if k != "randomSeed"}
        views = (r["public"] for r in source["rows"] if r["step"] <= TICKS[-1])
        tickets.append({**ident, **pick(views, cfg)})
    common.save(HERE / "roots.json", {"selector": sha(Path(__file__)), "tickets": tickets})
    print([(r["seed"], r.get("root_step"), r.get("pivot_role")) for r in tickets])


if __name__ == "__main__":
    main()

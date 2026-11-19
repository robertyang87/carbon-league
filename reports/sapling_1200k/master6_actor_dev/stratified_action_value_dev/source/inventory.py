"""Identity and use audit for historical raw tapes; no game outcomes enter map selection."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTORS = HERE.parents[1]
BASE = ACTORS.parent
ROOT = BASE.parents[1]
PARENT = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
EXAM = "57f9a594cce4f2ec9e3573d8d9d26ca1"
FROZEN = {"train": [621476120, 621476121, 621477120, 621477121],
          "held": [621461321, 621461322],
          "additional_held": [621475120, 621475121]}
EXTERNAL = {620224310, 620224311, 621454920, 621456920}
PRIOR_NATIVE_BUNDLE = 621497120


def checked(ok, why):
    if not ok:
        raise ValueError(why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def r(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def rel(path):
    return str(path.relative_to(ROOT))


def census():
    earlier = r(ACTORS / "late_local_residual_dev/protocol.json")
    forbidden = {x["map"] for x in earlier["source_groups"]}
    corridor = r(ACTORS / "late_harvest_corridor/protocol.json")
    proxy = {x["seed"] for g in corridor["groups"] for x in g["maps"]}
    used_roots = defaultdict(list)
    for p in sorted(BASE.rglob("live_protocol.json")):
        if HERE.parent in p.parents:
            continue
        c = r(p)
        if all(k in c for k in ("seed", "seat", "root_t")):
            used_roots[c["seed"], c["seat"]].append({"t": c["root_t"], "source": rel(p), "family": c.get("family")})
    t297 = {s["seed"] for b in r(ACTORS / "endgame_queue_dev/protocol.json")["source_batches"] for s in b["games"]}
    checked(all(isinstance(z, int) for z in t297), "malformed last-two map registry")
    groups = {seed:k for k,v in FROZEN.items() for seed in v}
    seen = Counter()
    count = Counter()
    cfgs = {}
    rows = []
    for p in sorted(BASE.rglob("raw_*.json.gz")):
        if HERE.parent in p.parents:
            continue
        tape = r(p)
        pack = tape.get("pack_sha256")
        count[pack] += 1
        if pack != PARENT or tape.get("examiner_md5") != EXAM:
            continue
        checked(not tape["errors"] and len(tape["actions"]) == tape["steps"]
                and [c[0] for c in tape["search_calls"]] == list(range(tape["steps"])), "incomplete game or original search callback")
        conf = tape["configuration"]
        cfg = tape["game_configuration"]
        checked(conf["_S_ENGINE"] is True and conf["_TEMP"] == 1.0 and conf["_SEED"] is None
                and conf["_S_HOMEFIX"] is True, "wrong opponent")
        phys = tape["candidate_seat"]
        seed = tape["seed"]
        checked(all(v == int(v) and 0 <= v <= 255 for v in tape["initial_full_carbon"]) and len(tape["initial_full_carbon"]) == 225, "opening format")
        terrain = digest(bytes(map(int, tape["initial_full_carbon"])))
        checked(terrain == tape["initial_terrain_u8_sha256"], "opening board mismatch")
        official = p.parent / "official_replay.json"
        cert = r(official)
        arm, = (a for a in cert["runs"].values() if a["pack_sha256"] == pack)
        game, = (g for g in arm["games"] if (g["seed"], g["seat"]) == (seed, phys))
        checked(game["transcript_sha256"] == digest(p.read_bytes()) and game["map_sha256"] == terrain
                and game["official_reward_by_seat"] == tape["official_score_by_seat"] and game["statuses"] == ["DONE", "DONE"]
                and game["steps"] == tape["steps"], "official statement mismatch")
        cfg_id = digest(json.dumps([cfg,conf], sort_keys=True, separators=(",",":")).encode())
        cfgs[cfg_id] = {"game_configuration": cfg, "examiner_configuration": conf}
        why = []
        if terrain in forbidden: why.append("previous late_local residual train/held map")
        if seed in proxy: why.append("late harvest corridor proxy panel")
        if seed in EXTERNAL or seed == PRIOR_NATIVE_BUNDLE: why.append("older native collector response panel")
        scope = groups.get(seed, "source_only")
        if len(why): checked(seed not in groups, "previous local/proxy terrain in proposed split")
        if not why and scope == "source_only" and seed == 621461220:
            why.append("previous global return-fit development trigger; avoid as primary held")
        pending = used_roots[seed,phys]
        exclude = {a["t"] for a in pending}
        if seed in t297: exclude.update((297,298))
        row = {"split":scope, "seed":seed, "seat":phys, "map":terrain, "raw":rel(p), "raw_sha256":digest(p.read_bytes()),
               "official":rel(official), "official_sha256":digest(official.read_bytes()), "config":cfg_id,
               "pack_sha256":pack, "examiner_md5":EXAM, "steps":tape["steps"],
               "previous_local_map":terrain in forbidden, "previous_corridor_map":seed in proxy,
               "known_raw_source_purpose":rel(p.parent), "other_label_root_sources":pending,
               "excluded_exact_physical_steps":sorted(exclude), "used_for_terminal_queue_t297_298":seed in t297,
               "selection_notes":why}
        rows.append(row)
        seen[seed,phys] += 1
    checked( len(set(r["map"] for r in rows)) == len({r["seed"] for r in rows}), "duplicate terrain across seeds")
    checked(all(seen[s,p] == 1 for s in groups for p in [0,1]), "planned map missing both unique full native seats")
    return {"schema":"actual-v1-native-raw-inventory-v1", "accepted_raw_pack_sha256":PARENT, "original_md5":EXAM,
            "all_raw_by_actor":dict(sorted(count.items())), "exact_v1_raw_count":len(rows),
            "exact_v1_map_count":len({r["map"] for r in rows}), "configurations":cfgs,
            "recommended":FROZEN, "games":[r for r in rows if r["split"] in ("train","held")],
            "optional_held_games":[r for r in rows if r["split"]=="additional_held"],"v1_inventory":rows,
            "exclusions":{"prior_local_map_hashes":sorted(forbidden), "prior_corridor_seeds":sorted(proxy),
            "other_native_collector_seeds":sorted(EXTERNAL | {PRIOR_NATIVE_BUNDLE})}}


if __name__ == "__main__":
    p = HERE / "inventory.json"
    if p.exists():
        out = r(p)
        for game in out["v1_inventory"]:
            checked(digest((ROOT/game["raw"]).read_bytes()) == game["raw_sha256"]
                    and digest((ROOT/game["official"]).read_bytes()) == game["official_sha256"],
                    "a frozen raw or official certificate changed")
    else:
        out = census()
        p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(out["exact_v1_raw_count"], out["exact_v1_map_count"], len(out["games"]))

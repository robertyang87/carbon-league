"""Count observed tactical opportunities in the frozen top-five head-to-head replays."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import ssl
import urllib.request
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TACTICS = ROOT / "reports/master_family/tactical_rules"
sys.path.insert(0, str(TACTICS / "detail"))
import cargo_defense
sys.path.insert(0, str(ROOT / "reports/master_family/deep_review"))
from recover_public import canonical, extract, key_of, validate
sys.path.insert(0, str(TACTICS))
from elite_defense_evidence import new_totals, observe as tree_events
from top5_defense import count_captures

REVISION = "499a8555c16d21eb56a78de1b8db1a1362b58f60"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def tactical_counts(replay, seat, key, trees):
    cells = cargo_defense.observe(replay, seat, key)["counts"]
    exposed = cells.get("stay_exposed_choose_safe", {})
    quiet = cells.get("home_no_enemy_all", {})
    ambush = [cells.get(name, {}) for name in ("chase_under30", "chase_30to100", "chase_100plus")]
    return {
        "intercept_opportunity": sum(x.get("n", 0) for x in ambush),
        "intercept_chosen": sum(x.get("selected", 0) for x in ambush),
        "intercept_confirmed": sum(x.get("prey_removed_and_hunter_survived", 0) for x in ambush),
        "cargo_endangered": exposed.get("n", 0),
        "cargo_safe_landing_chosen": exposed.get("selected", 0),
        "heavy_clear_home_opportunity": quiet.get("n", 0),
        "heavy_clear_home_chosen": quiet.get("selected", 0),
        "enemy_on_own_tree": trees["enemy_on_tree"],
        "friendly_contest": trees["ally_contest"],
        "tree_captured": trees["capture"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", required=True, type=Path)
    p.add_argument("--recent-replays", type=Path, help="local gzip directory for the seven later matches")
    p.add_argument("--fetch-public", action="store_true", help="allow read-only public spectator replay GETs for later missing blobs")
    p.add_argument("--insecure", action="store_true", help="allow the spectator IP's TLS hostname mismatch")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    reports = {k: (TACTICS / (k + ".json")).read_bytes()
               for k in ("top5_score_rows", "top5_inventory", "top5_defense")}
    scores, inventory, tree_report = (json.loads(reports[k]) for k in reports)
    if digest(reports["top5_score_rows"]) != inventory["score_rows_sha256"] or inventory["score_rows_sha256"] != tree_report["official_score_table_sha256"]:
        raise ValueError("score reports do not describe the same cohort")
    names = [r["name"] for r in inventory["ranks"]]
    ranks = {name: i + 1 for i, name in enumerate(names)}
    git_paths = defaultdict(list)
    files = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", REVISION, "data"], text=True)
    for path in files.splitlines():
        if path.endswith(".json.gz"):
            git_paths[Path(path).name[:-8]].append(path)
    historical = {k: v for k, v in scores.items() if v["name_a"] in ranks and v["name_b"] in ranks}
    if len(historical) != inventory["shared_fixture_count"]:
        raise ValueError("incomplete inter-top-five historical score census")
    registered = {}
    score_hashes = {}
    for handle in ("master5s", "master6", "it720preview", "m9xj630", "fablev7"):
        score_path = "data/" + handle + "_matches.json"
        payload = subprocess.check_output(["git", "-C", str(ROOT), "show", REVISION + ":" + score_path])
        score_hashes[score_path] = digest(payload)
        for row in json.loads(payload):
            if row.get("state") != "done" or row["name_a"] not in ranks or row["name_b"] not in ranks:
                continue
            key = key_of(row)
            prior = registered.get(key, historical.get(key))
            if prior and any(prior[f] != row[f] for f in ("name_a", "name_b", "scores", "order")):
                raise ValueError("conflicting official score/seat identity: " + key)
            registered[key] = row
    if not set(historical) <= registered.keys():
        raise ValueError("latest score files missing historical pair(s)")
    games = {}
    for key, match in sorted(registered.items()):
        seats = [match["name_" + side] for side in match["order"]]
        url = "https://117.72.210.134/replay/" + match["id"] + "/" + str(match["game"])
        path = None
        if key in historical:
            src = inventory["replays"][key]
            raw = (args.replays / (key + ".json.gz")).read_bytes()
            replay = json.loads(gzip.decompress(raw))
            expected = src["canonical_sha256"]
            original_source = src["source"]
        elif git_paths[key]:
            path = git_paths[key][0]
            raw = subprocess.check_output(["git", "-C", str(ROOT), "show", REVISION + ":" + path])
            replay = json.loads(gzip.decompress(raw))
            original_source, expected = "git", None
        else:
            local = args.recent_replays / (key + ".json.gz") if args.recent_replays else None
            if local is not None and local.is_file():
                raw = local.read_bytes()
                replay = json.loads(gzip.decompress(raw))
            elif args.fetch_public:
                request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
                ctx = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
                with urllib.request.urlopen(request, context=ctx, timeout=30) as response:
                    replay = extract(response.read())[0]
            else:
                raise ValueError("latest public raw missing: " + key)
            original_source, expected = "public", None
        canonical_hash = digest(canonical(replay))
        if expected and expected != canonical_hash:
            raise ValueError("historical canonical replay digest mismatch: " + key)
        check = validate(replay, match)
        if check["seat_model_names"] != seats or check["statuses"] != ["DONE", "DONE"]:
            raise ValueError("bad replay identities/status " + key)
        stats = []
        for seat, name in enumerate(seats):
            if key in historical:
                tref = tree_report["checked_games"][key]
                if tref["sha256"] != src["sha256"] or tref["seats"][str(seat)]["name"] != name:
                    raise ValueError("wrong tree report identity: " + key)
                trees = tref["seats"][str(seat)]
            else:
                trees = tree_events(replay["steps"], seat, replay["configuration"], name, key,
                                    seats[1-seat], new_totals(), {})
                if trees["capture"] != count_captures(replay["steps"], seat):
                    raise ValueError("tree ownership reconstruction disagrees: " + key)
            stats.append(tactical_counts(replay, seat, key, trees))
        games[key] = dict(ranks=[ranks[n] for n in seats], length=check["last_step"],
                          random_seed=check["random_seed"], env_uuid=check["environment_uuid"],
                          official_scores=check["rewards_by_seat"], is_later=key not in historical,
                          canonical_sha256=canonical_hash, original_source=original_source, public_url=url,
                          git_paths=git_paths[key], git_copies=len(git_paths[key]), by_seat=stats)
    if len({g["env_uuid"] for g in games.values()}) != len(games):
        raise ValueError("duplicate engine game under different match keys")
    def totals(group):
        group = list(group)
        sums, support = Counter(), Counter()
        for game in group:
            both = Counter()
            for side in game["by_seat"]:
                both.update(side)
            sums.update(both)
            support.update(k for k, n in both.items() if n)
        return dict(matches=len(group), joint_turns=sum(g["length"] for g in group),
                    archive_sources=dict(Counter(g["original_source"] for g in group)),
                    git_matches=sum(g["git_copies"] > 0 for g in group),
                    multiple_git_paths=sum(g["git_copies"] > 1 for g in group),
                    totals=dict(sums), contributing_matches={k:support[k] for k in sums})
    pairs = defaultdict(list)
    for game in games.values():
        pairs["-".join(map(str, sorted(game["ranks"])))].append(game)
    report = dict(
        revision=REVISION, ranks=names, selection="Keep the 2026-11-18 heavy-board ranks fixed. Audit all completed mutual games in the five official score-table snapshots at revision, including seven games beyond the original 101.",
        original_snapshot=totals(g for g in games.values() if not g["is_later"]),
        latest_score_tables_sha256=score_hashes,
        denominators="Two recorded players per match. Counts are worker- or tree-turns; an opportunity can repeat across turns. Supporting games are unique IDs. Alternative attack and defense labels can coexist.",
        definitions={"intercept": "Empty collector adjacent to exactly one eligible loaded opposing collector on a non-base cell. Confirmed requires hunter entering its cell, victim staying then disappearing, hunter surviving.",
                     "cargo": "Loaded collector would risk lethal/tie collision if staying and has a nonlethal currently free alternative, away from enemy bases. Chosen counts any nonlethal landing.",
                     "heavy_clear_home": "Quiet comparison local to the carrier: cargo >=100, sole adjacent free friendly base, alternate safe landing, no enemy can reach the base; chosen means entering it, not necessarily a deposit.",
                     "tree": "Enemy planter standing on a non-expiring owned tree; friendly contest is a friendly worker targeting that square. Captures checked using ownership transitions."},
        input_sha256={k:digest(v) for k,v in reports.items()},
        cargo_code_sha256=digest(Path(cargo_defense.__file__).read_bytes()),
        overall=totals(games.values()), pairs={p:totals(g) for p,g in sorted(pairs.items())}, games=games)
    with args.out.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["overall"]), flush=True)


if __name__ == "__main__":
    main()

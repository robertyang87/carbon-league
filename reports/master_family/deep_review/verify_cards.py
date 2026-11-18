"""Independent read-only acceptance of a generated Carbon per-game dossier.

From the repository root:
  python reports/master_family/deep_review/verify_cards.py --cache .cache/carbon-public
Add --out reports/master_family/deep_review/inputs/card_verification.json to save
the compact result. --repo and --directory override discovered defaults.
Checks Markdown cards/anchors/index/CSV, then verifies card scores, seats, final
steps, snapshots and selected events directly against replay bytes. It imports
neither review.py nor the event/spatial analysis functions.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

DEFAULT_REV = "37965ea578851b5f3ff88ffc42d7119ff716b5cc"
GAME = r"[0-9a-f]{32}-[0-9]+"
ANCHOR = re.compile(r'<a\s+id=[\"\']g-(' + GAME + r')[\"\']\s*>\s*</a>')
LINK = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
NUMBER = r"[+-]?[\d,]+(?:\.\d+)?"
MODELS = {"Master5s", "Master6"}
FAMILY = {"claude-AlphaCarbon-" + x for x in MODELS}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def number(text):
    return float(text.replace(",", ""))


def walk_matches(value):
    if isinstance(value, dict):
        if all(k in value for k in ("id", "game", "name_a", "name_b", "scores", "order")):
            yield value
        else:
            for child in value.values():
                yield from walk_matches(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_matches(child)


def observation(frame):
    for item in frame:
        obs = item.get("observation", {})
        if "players" in obs and "step" in obs:
            return obs
    raise ValueError("missing public observation")


def torus(a, b):
    ar, ac = divmod(int(a), 15)
    br, bc = divmod(int(b), 15)
    dr, dc = abs(ar - br), abs(ac - bc)
    return min(dr, 15 - dr) + min(dc, 15 - dc)


def independent_events(steps, seat):
    first_zone = first_near = first_tree = None
    full = [None, None]
    cash = []
    for t, frame in enumerate(steps):
        obs = observation(frame)
        assert obs["step"] == t, ("noncontiguous step", t, obs["step"])
        ps = obs["players"]
        cash.append(float(ps[seat][0]) - float(ps[1-seat][0]))
        for p in (0, 1):
            if len(ps[p][2]) == 10 and full[p] is None:
                full[p] = t
        own, enemy = ps[seat], ps[1-seat]
        own_bases, enemy_bases = list(own[1].values()), list(enemy[1].values())
        for _, (pos, _, _) in own[2].items():
            if not own_bases or not enemy_bases:
                continue
            do = min(torus(pos, b) for b in own_bases)
            de = min(torus(pos, b) for b in enemy_bases)
            if first_zone is None and de < do:
                first_zone = [t, int(pos)]
            if first_near is None and de <= 2:
                first_near = [t, int(pos)]
        if own[3] and first_tree is None:
            first_tree = [t, int(next(iter(own[3].values()))[0])]
    nonzero = [x for x in cash if x != 0]
    turns = sum((a > 0) != (b > 0) for a, b in zip(nonzero, nonzero[1:]))
    sign = (cash[-1] > 0) - (cash[-1] < 0)
    stable = max((t for t, x in enumerate(cash) if sign*x <= 0), default=-1) + 1 if sign else None
    return dict(first_zone=first_zone, first_near=first_near, first_tree=first_tree,
                first_full=[full[seat], full[1-seat]], cash=cash, turns=turns, stable=stable)


def validate(repo, directory, rev, limit=None, replay_cache=None):
    errors, warnings = [], []
    cards = {}
    markdown = sorted(directory.rglob("*.md"))
    if not markdown:
        raise ValueError(f"no generated Markdown under {directory}")
    for path in markdown:
        text = path.read_text()
        markers = list(ANCHOR.finditer(text))
        for i, match in enumerate(markers):
            key = match.group(1)
            if key in cards:
                errors.append(f"duplicate game card {key}: {cards[key]['path']} and {path}")
            cards[key] = dict(path=path, text=text[match.end():markers[i+1].start() if i+1<len(markers) else len(text)])
    if not cards:
        errors.append("no game cards parsed from explicit g-<match>-<game> anchors")

    # No private executor paths may escape into the published evidence/text.
    for path in directory.rglob("*"):
        if path.suffix.lower() not in (".md", ".json", ".jsonl", ".csv", ".py", ".txt") or not path.is_file():
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"(?<![A-Za-z0-9])/(?:tmp|workspace|home/agent|Users)/", line):
                errors.append(f"temporary/local path: {path.relative_to(directory)}:{n}")

    index = directory / "INDEX.md"
    if not index.is_file():
        errors.append("INDEX.md missing")
    index_counts = Counter()
    for path in markdown:
        text = path.read_text()
        for label, target in LINK.findall(text):
            target = unquote(target.strip().strip("<>"))
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("mailto:"):
                continue
            target_path = (path.parent / parsed.path).resolve() if parsed.path else path.resolve()
            if not target_path.exists():
                errors.append(f"missing local link in {path.name}: {target}")
                continue
            if parsed.fragment.startswith("g-"):
                key = parsed.fragment[2:]
                if key not in cards or cards[key]["path"].resolve() != target_path:
                    errors.append(f"wrong game anchor target in {path.name}: {target}")
                if re.fullmatch(GAME, label) and label != key:
                    errors.append(f"game link label/anchor disagree: {label} -> {key}")
                if path == index:
                    index_counts[key] += 1
    for key in cards:
        if index_counts[key] != 1:
            errors.append(f"INDEX.md must link game exactly once: {key}, links={index_counts[key]}")

    csv_path = directory / "all_games.csv"
    csv_rows, missing_keys, csv_by_key = [], [], {}
    if csv_path.is_file():
        csv_rows = list(csv.DictReader(csv_path.open(newline="")))
        csv_keys, analyzed_keys = [], []
        for row in csv_rows:
            key = row.get("key") or row.get("game_key")
            if not key and row.get("id") and row.get("game"):
                key = f"{row['id']}-{row['game']}"
            if key:
                csv_keys.append(key)
                csv_by_key[key] = row
                if row.get("model") not in MODELS:
                    errors.append(f"out-of-scope CSV focus model: {key}: {row.get('model')}")
                if row.get("replay_status", "analyzed") == "analyzed":
                    analyzed_keys.append(key)
                else:
                    missing_keys.append(key)
        if len(csv_keys) != len(csv_rows):
            errors.append("CSV rows do not all expose a full unique game key")
        if len(csv_keys) != len(set(csv_keys)):
            errors.append("duplicate game keys in all_games.csv")
        if set(cards) != set(analyzed_keys):
            errors.append(f"CSV analyzed/card set mismatch: cards_only={sorted(set(cards)-set(analyzed_keys))}, analyzed_only={sorted(set(analyzed_keys)-set(cards))}")
        if missing_keys:
            warnings.append(f"{len(missing_keys)} CSV games still lack analyzed replay cards; coverage is not complete")
    else:
        errors.append("all_games.csv missing")
    summary = {}
    if not (directory / "summary.json").is_file():
        errors.append("summary.json missing")
    else:
        summary = json.loads((directory / "summary.json").read_text())
        if set(summary.get("models_in_scope", [])) != MODELS:
            errors.append("summary scope must be exactly Master5s and Master6")
        if summary.get("metadata_games") != len(csv_rows) or summary.get("analyzed_games") != len(cards):
            errors.append("summary game counts disagree with CSV/card counts")
        if summary.get("unique_index_keys") != len(index_counts):
            errors.append("summary unique index count disagrees with Markdown")
        declared_pages = summary.get("pages", [])
        actual_sizes = [sum(c["path"].resolve() == (directory/p).resolve() for c in cards.values()) for p in declared_pages]
        if actual_sizes != summary.get("page_sizes") or sum(actual_sizes) != len(cards):
            errors.append("summary page sizes disagree with actual cards")

    # Build an independent match index from every frozen *_matches.json table.
    paths = git(repo, "ls-tree", "-r", "--name-only", rev).decode().splitlines()
    matches = defaultdict(list)
    replay_aliases = defaultdict(list)
    for path in paths:
        basename = Path(path).name
        m = re.fullmatch("(" + GAME + r")\.json(?:\.gz)?", basename)
        if m:
            replay_aliases[m.group(1)].append(path)
        if "matches" in basename and basename.endswith(".json"):
            try:
                rows = json.loads(git(repo, "show", f"{rev}:{path}"))
            except (ValueError, subprocess.CalledProcessError):
                continue
            for row in walk_matches(rows):
                key = f"{row['id']}-{row['game']}"
                if key in cards:
                    matches[key].append(row)
    official = directory / "inputs/official_matches.json"
    if official.is_file():
        for row in walk_matches(json.loads(official.read_text())):
            key = f"{row['id']}-{row['game']}"
            if key in cards:
                matches[key].append(row)

    checked = []
    for key, card in sorted(cards.items())[:limit]:
        try:
            text = card["text"]
            seat_match = re.search(r"关注方执席\s*([01])", text)
            score_match = re.search(r"比分\s*(" + NUMBER + r")\s*/\s*(" + NUMBER + r")", text)
            heading = re.search(r"^###\s+(.*?)\s+对\s+(.*?)\s*·\s*([胜负平])\s*(" + NUMBER + r")", text, re.M)
            assert seat_match and score_match and heading, "missing card seat/score/title fields"
            seat = int(seat_match.group(1))
            scores = [number(score_match.group(1)), number(score_match.group(2))]
            model = heading.group(1)
            assert model in MODELS, f"out-of-scope focus model {model}"
            title_margin = number(heading.group(4))
            assert abs(scores[0] - scores[1] - title_margin) <= .011, "title margin contradicts card scores"
            row_options = matches.get(key, [])
            row = next((r for r in row_options if r["name_"+r["order"][seat]].endswith(model)), None)
            assert row, "cannot map card model/seat to independent match table"
            names = [row["name_"+s] for s in row["order"]]
            expected = [float(row["scores"][row["order"][p]]) for p in (seat, 1-seat)]
            assert all(abs(a-b) <= .011 for a,b in zip(scores, expected)), "scores disagree with match table/order"
            csv_row = csv_by_key[key]
            assert csv_row["model"] == model and int(csv_row["seat"]) == seat, "CSV model/seat disagrees with card"
            assert all(abs(a-b) <= .011 for a,b in zip(scores,[float(csv_row['score']),float(csv_row['opponent_score'])])), "CSV/card score disagreement"
            assert abs(float(csv_row["margin"])-title_margin) <= .011, "CSV/card margin disagreement"
            assert csv_row["card"] == str(card["path"].relative_to(directory))+"#g-"+key, "CSV card target mismatch"
            assert heading.group(2).replace(r"\|", "|") == names[1-seat], "opponent label disagrees with match table"
            result = "胜" if expected[0] > expected[1] else "负" if expected[0] < expected[1] else "平"
            assert heading.group(3) == result, "win/loss label disagrees with scores"
            links = [url for label, url in LINK.findall(text) if label == "原始回放"]
            assert len(links) == 1, "exactly one raw-replay link required"
            link = links[0]
            assert csv_row["source_url"] == link, "CSV/card raw-source URL disagreement"
            source = link.split(f"/blob/{rev}/", 1)[1] if f"/blob/{rev}/" in link else None
            raw = None
            if source:
                try:
                    raw = git(repo, "show", f"{rev}:{source}")
                except subprocess.CalledProcessError:
                    pass
            if raw is None:
                local_candidates = [p for p in directory.rglob(f"{key}.json*") if p.is_file()]
                if replay_cache:
                    local_candidates += [p for p in replay_cache.glob(f"{key}.json*") if p.is_file()]
                if local_candidates:
                    raw = local_candidates[0].read_bytes()
                elif replay_aliases.get(key):
                    source = replay_aliases[key][0]
                    raw = git(repo, "show", f"{rev}:{source}")
            assert raw is not None, "no independently readable replay bytes for card"
            assert hashlib.sha256(raw).hexdigest() == csv_row['sha256'], "CSV replay SHA256 disagrees with bytes"
            payload = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
            replay = json.loads(payload)
            steps = replay["steps"]
            reward = [float(replay["rewards"][p]) for p in (seat, 1-seat)]
            assert all(abs(a-b) <= .011 for a,b in zip(reward, scores)), "replay official rewards disagree with card"
            ev = independent_events(steps, seat)
            last = len(steps)-1
            assert int(csv_row['last_step']) == last, "CSV final state step disagrees with replay"
            table = re.findall(r"^\|\s*(\d+)\s*\|\s*(" + NUMBER + r")\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|\s*(" + NUMBER + r")\s*/\s*(" + NUMBER + r")\s*\|", text, re.M)
            assert table and int(table[-1][0]) == last, "final snapshot missing or wrong final step"
            for t, gap, w0, w1, tr0, tr1, c0, c1 in table:
                t = int(t)
                ps = observation(steps[t])["players"]
                ix = (seat, 1-seat)
                assert abs(number(gap)-ev["cash"][t]) <= .011, f"snapshot cash mismatch t{t}"
                assert [int(w0),int(w1)] == [len(ps[p][2]) for p in ix], f"snapshot workers mismatch t{t}"
                assert [int(tr0),int(tr1)] == [len(ps[p][3]) for p in ix], f"snapshot trees mismatch t{t}"
                cargo = [sum(float(w[1]) for w in ps[p][2].values()) for p in ix]
                assert all(abs(a-b) <= .011 for a,b in zip(cargo,[number(c0),number(c1)])), f"snapshot cargo mismatch t{t}"
            for label, field in (("更靠近敌基地的区域为", "first_zone"), ("首次到敌基地两格以内为", "first_near"), ("首次持有树为", "first_tree")):
                m = re.search(re.escape(label)+r"\s*t(\d+)\s*格(\d+)", text)
                if ev[field] is not None:
                    assert m and [int(m.group(1)),int(m.group(2))] == ev[field], f"first-event mismatch {field}"
                else:
                    assert re.search(re.escape(label)+r"\s*未观察到", text), f"spurious first-event {field}"
            stable = re.search(r"末现金差方向从\s*t(\d+)\s*起", text)
            if ev["stable"] is not None:
                assert stable and int(stable.group(1)) == ev["stable"], "cash reversal/stability time mismatch"
            direct = all(n in FAMILY for n in names)
            late_reverse = last >= 250 and (ev["cash"][250] * (scores[0]-scores[1]) < 0)
            checked.append(dict(key=key, model=model, names=[names[seat],names[1-seat]], seat=seat, scores=scores, last=last,
                                snapshots=len(table), first_events_verified=3, stable=ev["stable"],
                                turns=ev["turns"], direct_family=direct,
                                late_reversal=late_reverse,
                                sha256=hashlib.sha256(raw).hexdigest()))
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    model_counts = {m: Counter() for m in MODELS}
    for game in checked:
        for i, name in enumerate(game['names']):
            if name in FAMILY:
                stats = model_counts[name.removeprefix('claude-AlphaCarbon-')]
                stats['games'] += 1
                a,b = game['scores'][i],game['scores'][1-i]
                stats['wins' if a>b else 'losses' if a<b else 'draws'] += 1
    if limit is None:
        for model, counts in model_counts.items():
            claimed = summary.get('models',{}).get(model,{}).get('analyzed',{})
            for field in ('games','wins','losses','draws'):
                if counts[field] != claimed.get(field):
                    errors.append(f"summary {model}/{field}: claimed={claimed.get(field)}, independently verified={counts[field]}")
    return {"revision": rev, "models_in_scope": sorted(MODELS), "cards": len(cards), "csv_rows": len(csv_rows),
            "missing_replay_keys": missing_keys, "complete_replay_coverage": not missing_keys,
            "index_game_links": sum(index_counts.values()), "verified_games": len(checked),
            "direct_cases": [x for x in checked if x["direct_family"]],
            "model_appearances": {m:dict(c) for m,c in model_counts.items()},
            "late_reversal_cases": [x for x in checked if x["late_reversal"]],
            "errors": errors, "warnings": warnings, "passed": not errors,
            "checks": checked}


def compact_result(result):
    """Keep validation evidence without copying every intermediate game row."""
    return {
        "schema": "carbon-independent-card-verification/1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision": result["revision"],
        "scope": result["scope"],
        "models_in_scope": result["models_in_scope"],
        "method": "Independent parser and direct replay-state calculations; no imports from the report generator or its event/spatial analyzers.",
        "dimensions": [
            "one Markdown card per full matchid-game, no duplicate anchors",
            "one CSV row and one correct INDEX link per analyzed game",
            "top-two-only focus models and summary/page-count reconciliation",
            "no executor-specific absolute paths in published text or scripts",
            "official score-table seat/order, score, opponent and win/loss labels",
            "CSV/card/source URL consistency and replay compressed-byte SHA256",
            "contiguous observation steps and actual final step, no end padding",
            "every displayed cash/worker/tree/cargo snapshot",
            "first enemy-zone entry, enemy-near entry and own-tree state",
            "last cash-sign reversal/stability step and direct-match uniqueness",
            "per-model appearances and outcome totals, including both direct-match seats",
        ],
        "counts": {
            "cards": result["cards"], "csv_rows": result["csv_rows"],
            "index_game_links": result["index_game_links"],
            "verified_games": result["verified_games"],
            "verified_snapshots": sum(g["snapshots"] for g in result["checks"]),
            "first_event_checks": sum(g["first_events_verified"] for g in result["checks"]),
            "direct_top2_games": len(result["direct_cases"]),
            "late_reversal_games": len(result["late_reversal_cases"]),
            "early_terminal_games": sum(g["last"] < 299 for g in result["checks"]),
        },
        "model_appearances": result["model_appearances"],
        "complete_replay_coverage": result["complete_replay_coverage"],
        "missing_replay_keys": result["missing_replay_keys"],
        "direct_top2_keys": [g["key"] for g in result["direct_cases"]],
        "late_reversal_definition": "The displayed focus side's cash gap at state 250 has the opposite sign from its official final score margin.",
        "late_reversal_keys": [g["key"] for g in result["late_reversal_cases"]],
        "early_terminal": [{"key": g["key"], "last_state": g["last"]}
                           for g in result["checks"] if g["last"] < 299],
        "errors": result["errors"], "warnings": result["warnings"],
        "passed": result["passed"],
    }


def discover_repo():
    for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    raise ValueError("cannot discover repository; provide --repo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="Defaults to the current or enclosing Git repository")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--rev", default=DEFAULT_REV)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache", "--replay-cache", dest="replay_cache", type=Path,
                        help="Optional local archive of publicly downloaded replay bytes")
    parser.add_argument("--limit", type=int, help="Smoke-test only this many replay bodies; omit for acceptance")
    args = parser.parse_args()
    args.repo = (args.repo or discover_repo()).resolve()
    directory = args.directory or args.repo / "reports/master_family/deep_review"
    result = validate(args.repo, directory, args.rev, args.limit, args.replay_cache)
    result["scope"] = "partial smoke" if args.limit is not None else "all parsed cards"
    compact = compact_result(result)
    if args.out:
        args.out.write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

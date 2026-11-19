"""Bind a completed native-corpus release before any window simulation is authorized."""
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
BASE = ROOT / "reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev"
CORPUS = BASE / "native_actor_corpus"
PROTOCOL = "1b23910b1d97755483fbdc1fb05550ae816abc5323b4402e2f2c480e0a727703"
PANEL = "fb490e72f2df1fac20a7c26310523e165956a5619fad7de46104604096611124"
FIT = "336c9f1a212387e313eb906233f206f692125cf762790153c305ac5f75ac72f2"
EVALUATE = "dab8e6584469028472f8f302f6c7b0123ba78bdf329bc00d34b5a9d64264edab"
ARMS = ("parent", "head", "full")
ROLES = ("collector", "planter")
SPLITS = ("val", "test")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sig(obj):
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def panel_cards():
    check(sha(CORPUS / "protocol.json") == PROTOCOL and sha(BASE / "panel.json") == PANEL, "registration changed")
    rule, panel = read(CORPUS / "protocol.json"), read(BASE / "panel.json")
    return [{**c, "split": "excluded" if c["seed"] in rule["quarantine"] else panel["split"][c["terrain"]]}
            for c in panel["cards"]]


def planned_cases():
    cards = [c for c in panel_cards() if c["split"] == "test"]
    seeds = sorted({c["seed"] for c in cards})
    check(len(seeds) == 6, "expected six held terrains")
    cases = []
    for seat, start in enumerate((64, 160)):
        card, = [c for c in cards if c["seed"] == seeds[seat] and c["seat"] == seat]
        cases.append({"seed": card["seed"], "terrain": card["terrain"], "seat": seat, "t0": start,
                      "t1_exclusive": start + 32, "source": card["path"],
                      "source_sha256": card["sha256"], "receipt": card["receipt_path"],
                      "receipt_sha256": card["receipt_sha256"],
                      "arm_order": ["parent", "student"] if seat == 0 else ["student", "parent"]})
    return cases


def verify_rows(rows, cards, reports):
    expected = {c["sha256"]: c for c in cards if c["split"] in SPLITS}
    steps, correct, count, changes = defaultdict(set), Counter(), Counter(), Counter()
    for r in rows:
        tape, t = r["tape_sha256"], r["step"]
        c = expected[tape]
        check((r["seed"], r["terrain"], r["split"], r["expert_seat"]) ==
              (c["seed"], c["terrain"], c["split"], 1 - c["seat"]), "held command membership mismatch")
        check(0 <= t < 299 and t not in steps[tape], "duplicate or invalid held step")
        steps[tape].add(t)
        terrain = c["terrain"]
        for uid, role in r["roles_by_uid"].items():
            if role not in ROLES:
                check(role == "base", "unrecognized owned entity")
                continue
            truth = r["expert"].get(uid) or "NONE"
            cmd = {a: r["deployed"][a].get(uid) or "NONE" for a in ARMS}
            changes[terrain] += cmd["full"] != cmd["parent"]
            for group in (role, "worker"):
                count[terrain, group] += 1
                for a in ARMS:
                    correct[terrain, group, a] += cmd[a] == truth
    check(set(steps) == set(expected) and all(len(v) == 24 for v in steps.values()), "incomplete held decisions")
    for split in SPLITS:
        maps = sorted({c["terrain"] for c in expected.values() if c["split"] == split})
        report = reports[split]
        check(set(report["by_terrain"]) == set(maps), "terrain summary mismatch")
        for m in maps:
            summary = report["by_terrain"][m]
            check(changes[m] == summary["pairs"]["full_minus_parent"]["worker"]["changed"], "changed count mismatch")
            for role in (*ROLES, "worker"):
                for a in ARMS:
                    item = summary["arms"][a][role]
                    check((item["entities"], item["correct"]) == (count[m, role], correct[m, role, a]),
                          "normal-entry command count mismatch")
        for a in ARMS:
            for role in (*ROLES, "worker"):
                rate = sum(correct[m, role, a] / count[m, role] for m in maps) / len(maps)
                check(abs(rate - report["map_equal_accuracy"][a][role]) < 1e-12, "map-equal accuracy mismatch")
    scores = reports["test"]["map_equal_accuracy"]
    test_maps = reports["test"]["by_terrain"]
    check(scores["full"]["worker"] > max(scores[a]["worker"] for a in ("head", "parent"))
          and sum(changes[m] > 0 for m in test_maps) >= 2
          and all(scores["full"][r] - scores["parent"][r] >= -0.02 for r in ROLES),
          "full backbone failed frozen free-decoding gate")


def freeze(candidate_sha, split_sha, val_sha, test_sha):
    """split_sha signs sources.json bytes. val_sha and test_sha sign sig(held['results'][split])."""
    cards = panel_cards()
    rule = read(CORPUS / "protocol.json")
    check(sha(CORPUS / "sources.json") == split_sha and sha(CORPUS / "fit.py") == FIT
          and sha(CORPUS / "evaluate.py") == EVALUATE, "corpus or reviewed source mismatch")
    members = {c["sha256"]: (c["seed"], c["terrain"], c["seat"], c["split"]) for c in cards}
    source = read(CORPUS / "sources.json")["cards"]
    for c in source:
        check(members[c["sha256"]] == (c["seed"], c["terrain"], c["seat"], c["split"])
              and c["expert_seat"] == 1 - c["seat"], "source card identity mismatch")
    included = [c["sha256"] for c in source if c["split"] != "excluded"]
    check(sorted(included) == sorted(h for h, k in members.items() if k[-1] != "excluded"), "incomplete split")
    fit = read(CORPUS / "fit.json")
    check(fit["parent"] == rule["parent_sha256"] and fit["weights"] == rule["weights_sha256"],
          "initial weights or parent mismatch")
    for name, digest in fit["signature"].items():
        check(sha(CORPUS / name) == digest, "fit input mismatch: " + name)
    packages = {"parent": sha(BASE / "candidate.py")}
    for arm in ("head", "full"):
        entry = fit["arms"][arm]
        packages[arm] = sha(CORPUS / (arm + ".py"))
        check(packages[arm] == entry["package_sha256"] and sha(CORPUS / (arm + ".npz")) == entry["weights_sha256"]
              and entry["updates"] == 312 and entry["unique_roots"] == 2496, "final release mismatch: " + arm)
    check(packages["full"] == candidate_sha and packages["parent"] == rule["parent_sha256"], "arm SHA mismatch")
    held = read(CORPUS / "held.json")
    held_path = CORPUS / "held_commands.json.gz"
    check(sha(held_path) == held["held_commands_sha256"], "command ledger SHA mismatch")
    ledger = read(held_path)
    observed = {n: sha(CORPUS / n) for n in ("protocol.json", "sources.json", "fit.json", "common.py", "evaluate.py")}
    check(observed == ledger["source_sha256"] == held["source_sha256"]
          and packages == ledger["package_sha256"] == held["package_sha256"], "held binding mismatch")
    reports = held["results"]
    check([sig(reports[s]) for s in SPLITS] == [val_sha, test_sha], "signed split result mismatch")
    verify_rows(ledger["rows"], cards, reports)
    check(held["qualification"]["full"]["passed"] is True, "published full-arm gate disagrees")
    cases = planned_cases()
    for case in cases:
        check(sha(ROOT / case["source"]) == case["source_sha256"] and sha(ROOT / case["receipt"]) == case["receipt_sha256"],
              "historical transcript or receipt changed")
    paths = {"parent": BASE / "candidate.py", **{a: CORPUS / (a + ".py") for a in ("head", "full")}}
    return {"packages": {a: {"path": str(p.relative_to(ROOT)), "sha256": packages[a]} for a, p in paths.items()},
            "sources_sha256": split_sha, "protocol_sha256": PROTOCOL,
            "val_sha256": val_sha, "test_sha256": test_sha, "held_sha256": sha(CORPUS / "held.json"),
            "held_commands_sha256": held["held_commands_sha256"], "cases": cases}


def revalidate(plan):
    bound = freeze(plan["packages"]["full"]["sha256"], plan["sources_sha256"], plan["val_sha256"], plan["test_sha256"])
    check(bound == plan, "response manifest differs from frozen evidence")
    return bound

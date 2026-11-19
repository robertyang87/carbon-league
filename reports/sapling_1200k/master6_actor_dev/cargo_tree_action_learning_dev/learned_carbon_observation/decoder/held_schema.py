"""Join signed historical commands and summarize held-trajectory decoder ablations."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ESTIMATOR = HERE.parent
ROOT = HERE.parents[5]
ROLES = ("collector", "planter", "base")
NEW_ARMS = ("decoder_student", "decoder_constant")
COMPARISONS = (
    ("incumbent", "decoder_student"),
    ("encoder_student", "decoder_student"),
    ("decoder_constant", "decoder_student"),
    ("incumbent", "decoder_constant"),
    ("encoder_constant", "decoder_constant"),
    ("encoder_student", "decoder_constant"),
    ("encoder_constant", "decoder_student"),
    ("incumbent", "encoder_student"),
)


def _read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_references():
    """Return ({(seed, seat, step): commands by arm}, signed source identities)."""
    protocol_path = HERE / "protocol.json"
    estimator_path = ESTIMATOR / "protocol.json"
    decoder = _read(protocol_path)
    estimator = _read(estimator_path)
    cards = [c for c in estimator["cards"] if c["split"] == "val"]
    assert len(cards) == len({c["seed"] for c in cards}) == 3
    identities = {
        "incumbent": decoder["base"],
        "encoder_student": decoder["encoder_student"],
        "encoder_constant": decoder["encoder_constant"],
    }
    assert identities["incumbent"] == estimator["incumbent"]
    for arm, path in (("incumbent", ESTIMATOR.parent / "candidate.py"),
                      ("encoder_student", ESTIMATOR / "candidate.py"),
                      ("encoder_constant", ESTIMATOR / "constant.py")):
        assert _sha(path) == identities[arm], (arm, "reference package SHA")

    for c in cards:
        path = ROOT / c["path"]
        assert _sha(path) == c["sha256"], (c["seed"], "source tape SHA")
        game = _read(path)
        actual = hashlib.sha256(bytes(int(n) for n in game["initial_full_carbon"])).hexdigest()
        assert actual == game["initial_terrain_u8_sha256"] == c["terrain"]
        assert len(game["actions"]) == 299 and game["seed"] == c["seed"]

    first_summary = _read(ESTIMATOR / "normal_summary.json")
    normal_path = ESTIMATOR / "normal_entry.json.gz"
    assert first_summary["card"] == cards[0]
    assert first_summary["protocol_sha256"] == _sha(estimator_path)
    assert first_summary["detail_sha256"] == _sha(normal_path)
    normal = _read(normal_path)
    refs = {}
    for seat in (0, 1):
        for arm, old_name in (("incumbent", "incumbent"), ("encoder_student", "student"),
                              ("encoder_constant", "constant")):
            pack = normal[old_name][seat]
            assert pack["timing"]["ready"]["pack_sha256"] == identities[arm]
            assert first_summary["timing"][old_name]["pack_sha256"] == identities[arm]
            assert len(pack["timing"]["rows"]) == len(pack["audit"]["rows"]) == 299
            for t, (plain, traced) in enumerate(zip(pack["timing"]["rows"], pack["audit"]["rows"], strict=True)):
                assert plain["action"] == traced["action"]
                refs.setdefault((cards[0]["seed"], seat, t), {})[arm] = plain["action"]

    other_summary = _read(ESTIMATOR / "other_held_actions.json")
    other_path = ESTIMATOR / "other_held_action_tapes.json.gz"
    assert other_summary["protocol_sha256"] == _sha(estimator_path)
    assert other_summary["tapes_sha256"] == _sha(other_path)
    assert [r["card"] for r in other_summary["records"]] == cards[1:]
    assert other_summary["pack_sha256"] == {"incumbent": identities["incumbent"], "student": identities["encoder_student"]}
    for r in _read(other_path):
        key = r["seed"], r["seat"], r["t"]
        assert key not in refs
        refs[key] = dict(incumbent=r["parent"], encoder_student=r["student"])
    expected = {(c["seed"], seat, t) for c in cards for seat in (0, 1) for t in range(299)}
    assert set(refs) == expected
    source = dict(
        protocol_sha256=_sha(protocol_path), estimator_protocol_sha256=_sha(estimator_path), packages=identities,
        first_detail_sha256=_sha(normal_path), first_manifest_sha256=_sha(ESTIMATOR / "normal_summary.json"),
        other_detail_sha256=_sha(other_path), other_manifest_sha256=_sha(ESTIMATOR / "other_held_actions.json"),
        archived_physical_trajectories=cards,
        encoder_constant_coverage="Archived for both seats on the first of three terrains.",
    )
    return refs, source


def _measure_changes(rows, left, right):
    total = Counter()
    by_role = {role: Counter() for role in ROLES}
    for row in rows:
        if left not in row["commands"] or right not in row["commands"]:
            continue
        a, b = row["commands"][left], row["commands"][right]
        roles = row["roles"]
        assert set(a) | set(b) <= set(roles)
        assert set(roles.values()) <= set(ROLES)
        exposed_roles = set(roles.values())
        changed_roles = set()
        total["turns"] += 1
        total["slots"] += len(roles)
        for uid, role in roles.items():
            by_role[role]["slots"] += 1
            if a.get(uid, "NONE") != b.get(uid, "NONE"):
                by_role[role]["changed_slots"] += 1
                total["changed_slots"] += 1
                changed_roles.add(role)
        for role in exposed_roles:
            by_role[role]["turns"] += 1
            by_role[role]["changed_turns"] += int(role in changed_roles)
        total["changed_turns"] += int(bool(changed_roles))
    def normalized(counter):
        return {k: int(counter[k]) for k in ("turns", "changed_turns", "slots", "changed_slots")}
    return {**normalized(total), "by_role": {k: normalized(v) for k, v in by_role.items()}}


def _distribution(values):
    a = np.asarray(values, dtype=np.float64)
    assert len(a) and np.isfinite(a).all()
    return {**{f"p{q}": round(float(np.percentile(a, q)), 4) for q in (50, 95, 99)},
            "mean": round(float(a.mean()), 4), "max": round(float(a.max()), 4)}


def _timing(rows, arm):
    timed = [r["timing"][arm] for r in rows]
    return dict(calls=len(timed), entry_ms=_distribution([t["entry_ms"] for t in timed]),
                external_ms=_distribution([t["external_ms"] for t in timed]),
                cold_first_call_external_ms=[t["external_ms"] for t, r in zip(timed, rows) if r["t"] == 0],
                external_over_1s=sum(t["external_ms"] > 1000 for t in timed),
                external_over_3s=sum(t["external_ms"] > 3000 for t in timed),
                peak_rss_kib=max(t["peak_rss_kib"] for t in timed))


def _audit_counts(rows, arm):
    counts = Counter()
    for row in rows:
        counts.update(row["audit"][arm])
    return dict(counts)


def _group(rows):
    comparisons = {f"{left}_to_{right}": _measure_changes(rows, left, right) for left, right in COMPARISONS
                   if any(left in r["commands"] and right in r["commands"] for r in rows)}
    return dict(comparisons=comparisons,
                timing={name: _timing(rows, name) for name in NEW_ARMS},
                audit_counter_sums={name: _audit_counts(rows, name) for name in NEW_ARMS})


def summarize(rows, meta):
    """Enrich commands in place and return a JSON-ready source-backed summary."""
    refs, source = load_references()
    assert len(rows) == len(refs)
    visited = set()
    for row in rows:
        key = row["seed"], row["seat"], row["t"]
        assert key not in visited and key in refs
        visited.add(key)
        assert set(row["commands"]) == set(NEW_ARMS)
        row["commands"].update(refs[key])
    assert visited == set(refs)
    maps = {}
    for card in source["archived_physical_trajectories"]:
        seed = card["seed"]
        subset = [r for r in rows if r["seed"] == seed]
        maps[str(seed)] = {**_group(subset), "terrain": card["terrain"],
                           "by_physical_seat": {str(s): _group([r for r in subset if r["seat"] == s])
                                                for s in (0, 1)}}
    return dict(source=source, run=meta, fixed_trajectory_calls_per_arm=len(rows), overall=_group(rows), by_map=maps,
                denominators="Turns are seat-specific observations. Per-role turns count observations with at least one live asset of that role; slots count every live controllable asset. Absent commands select NONE. Missing encoder-constant reference on two terrains is excluded from both numerator and denominator.",
                timing_scope="Independent uninstrumented public entry. Cold first-call times appear in the full distribution and separately. Worker process startups are recorded in run metadata.")

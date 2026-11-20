"""Exact likelihood on sampled physical histories through their first zero for the historical student update."""

import argparse
import gzip
import json
import math
import sys
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[1]
ROOT = ACTOR.parents[2]
INTAKE = (
    ACTOR / "cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake"
)
sys.path.insert(0, str(INTAKE))
from replay_actions import audit_menu, distribution, order_key, probabilities

MANIFEST = INTAKE / "hosted/training_manifest.json"
PARENT = audit_menu.PACKS["candidate"][0]
CHILD = INTAKE / "candidate.py"


def need(flag, reason):
    if not flag:
        raise ValueError(reason)


def hash_file(p):
    return sha256(p.read_bytes()).hexdigest()


def view(p):
    return [{"action": dict(k), "probability": v} for k, v in sorted(p.items())]


def physical_likelihood(old, new, sent):
    """Support of executed orders, including unit identities, not slot indices or raw logit labels.

    None in a log field represents minus infinity; the exact ratio remains numeric zero.
    Positive target mass outside the behavior support is an additional, opposite-direction issue.
    """
    key = order_key(sent)
    p = old.get(key, 0.0)
    q = new.get(key, 0.0)
    need(
        math.isclose(sum(old.values()), 1.0, abs_tol=1e-12)
        and math.isclose(sum(new.values()), 1.0, abs_tol=1e-12)
        and all(math.isfinite(v) and v > 0 for v in [*old.values(), *new.values()]),
        "Invalid physical probabilities",
    )
    need(p > 0, "Logged action impossible under its alleged behavior policy")
    absent = sum(v for k, v in old.items() if k not in new)
    novel = sum(v for k, v in new.items() if k not in old)
    return {
        "old": p,
        "new": q,
        "ratio": q / p,
        "log_old": math.log(p),
        "log_new": math.log(q) if q else None,
        "log_ratio": math.log(q / p) if q else None,
        "old_mass_missing_in_new": absent,
        "new_mass_outside_old": novel,
    }


def official_prefix(ep):
    source = ROOT / ep["archive_path"]
    need(hash_file(source) == ep["archive_sha256"], "Wrong archived episode ZIP")
    with zipfile.ZipFile(source) as z:
        registration = z.read(ep["registration_member"])
        receipt = z.read(ep["receipt_member"])
        tape = z.read(ep["turns_member"])
    need(
        sha256(registration).hexdigest() == ep["registration_sha256"]
        and sha256(receipt).hexdigest() == ep["receipt_sha256"]
        and sha256(tape).hexdigest() == ep["tape_sha256"] == json.loads(receipt)["turns_sha256"],
        "Archived source identity failed",
    )
    return registration, tape.splitlines()


def measure(ep):
    seat = ep["seat"]
    sites = {v["step"]: v for v in ep["events"]}
    reg, tape = official_prefix(ep)
    n = ep["official_transitions"]
    need(
        n == len(tape) and set(sites) == set(audit_menu.STEPS) and len(sites) == len(ep["events"]),
        "Incomplete original batch",
    )
    old = audit_menu.collect.Player(PARENT)
    new = audit_menu.collect.Player(CHILD)
    running = 0.0
    seen = []
    first_zero = None
    first_new = None
    max_old_gap = 0.0
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / "registration.json"
        p.write_bytes(reg)
        live = audit_menu.open_live(p)
        env, cfg = live.official()
        bank = [5.0, 5.0]
        need(
            sha256(bytes(env.states[0].observation.full_carbon)).hexdigest() == ep["terrain"],
            "Opening terrain changed",
        )
        for t, raw in enumerate(tape):
            tr = json.loads(raw)
            obs = live.observe(env, seat, bank[seat])
            dg = sha256(live.encoded(obs)).hexdigest()
            need(
                not env.done and obs["step"] == t and dg == tr["public_observation_sha256"][seat],
                "Recorded pre-action input differs",
            )
            actions = [call["response"]["action"] for call in tr["calls"]]
            _, p0 = distribution(old, obs, cfg)
            _, p1 = distribution(new, obs, cfg)
            m = physical_likelihood(p0, p1, actions[seat])
            example = dict(
                t=t,
                public_sha256=dg,
                scheduled=t in sites,
                issued=actions[seat],
                old=view(p0),
                new=view(p1),
                **{k: v for k, v in m.items() if k not in ("old", "new")},
            )
            if t in sites:
                event = sites[t]
                need(
                    order_key(event["issued"]) == order_key(actions[seat])
                    and probabilities(event["options"], event["probabilities"]) == p0
                    and event["public_observation_sha256"] == dg,
                    "Logged action or behavior support differs",
                )
                gap = abs(event["logp"] - m["log_old"])
                need(gap < 1e-10, "Old physical log likelihood drift")
                max_old_gap = max(max_old_gap, gap)
            else:
                need(
                    len(p0) == len(p1) == 1 and p0[order_key(actions[seat])] == 1.0,
                    "Unscheduled behavior is not deterministic",
                )
            seen.append(
                {
                    "t": t,
                    "public_sha256": dg,
                    "logp_old": m["log_old"],
                    "logp_new": m["log_new"],
                    "ratio": m["ratio"],
                    "mass_outside_behavior": m["new_mass_outside_old"],
                    "scheduled": t in sites,
                }
            )
            if first_new is None and m["new_mass_outside_old"] > 0:
                first_new = example
            if m["ratio"] == 0:
                first_zero = example
                break
            running += m["log_ratio"]
            bank = tr["overage_remaining_seconds"]
            env.step(actions)
        if first_zero is None:
            need(
                env.done and [s.reward for s in env.states] == ep["official_scores_by_seat"],
                "Nonzero episode incomplete",
            )
    return {
        "seed": ep["seed"],
        "seat": seat,
        "arm": ep["arm"],
        "terrain": ep["terrain"],
        "source_sha": ep["archive_sha256"],
        "turns_sha": ep["tape_sha256"],
        "full_recorded_turns": n,
        "checked": len(seen),
        "scheduled_checked": sum(r["scheduled"] for r in seen),
        "max_old_logp_error": max_old_gap,
        "first_zero": first_zero,
        "first_target_novel": first_new,
        "last_nonzero_prefix_log_ratio": running,
        "whole_log_ratio": None if first_zero is not None else running,
        "whole_ratio": 0.0 if first_zero is not None else math.exp(running),
        "history": seen,
    }


def save(p, d):
    v = (
        json.dumps(d, sort_keys=True, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    if p.suffix == ".gz":
        v = gzip.compress(v, mtime=0)
    if p.exists():
        need(p.read_bytes() == v, "Existing output mismatch " + str(p))
    else:
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(v)
        tmp.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--require-recorded-overlap",
        action="store_true",
        help="Exit nonzero on witnessed target-only support or all-zero trajectory weights; applies to trajectory reweighting only.",
    )
    a = parser.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    d = json.loads(MANIFEST.read_bytes())
    fit = json.loads((INTAKE / "fit.json").read_bytes())
    need(
        hash_file(PARENT) == d["policy_sha256"] == fit["parent_sha256"]
        and hash_file(CHILD) == fit["selected_package_sha256"]
        and hash_file(MANIFEST) == fit["archive_manifest_sha256"]
        and fit["selected_step"] == 0.1,
        "Historical trainable package identity",
    )
    episodes = d["episodes"]
    idset = {(e["seed"], e["seat"], e["arm"]) for e in episodes}
    need(
        len(episodes) == len(idset) == 24
        and len({e["terrain"] for e in episodes}) == 6
        and all(e["episode_verified"] for e in episodes),
        "Wrong natural batch",
    )
    rows = []
    for e in episodes:
        p = a.work / f"{e['seed']}_{e['seat']}_{e['arm']}.json.gz"
        if p.exists():
            record = json.loads(gzip.decompress(p.read_bytes()))
            need(
                record["source_sha"] == e["archive_sha256"]
                and record["turns_sha"] == e["tape_sha256"]
                and record["method"] == hash_file(Path(__file__))
                and record["old_package"] == hash_file(PARENT)
                and record["new_package"] == hash_file(CHILD),
                "Cached prefix no longer matches",
            )
        else:
            record = measure(e)
            record.update(
                method=hash_file(Path(__file__)),
                old_package=hash_file(PARENT),
                new_package=hash_file(CHILD),
            )
            save(p, record)
        rows.append(record)
        print(
            len(rows),
            e["seed"],
            e["seat"],
            e["arm"],
            record["checked"],
            record["whole_ratio"],
            flush=True,
        )
    save(HERE / "prefixes.json.gz", rows)
    support = [r for r in rows if r["whole_ratio"] > 0]
    ratios = [r["whole_ratio"] for r in rows]
    result = {
        "parent": hash_file(PARENT),
        "child": hash_file(CHILD),
        "manifest": hash_file(MANIFEST),
        "prefixes": hash_file(HERE / "prefixes.json.gz"),
        "method": hash_file(Path(__file__)),
        "episodes": len(rows),
        "maps": len({r["terrain"] for r in rows}),
        "recorded_turns": sum(r["full_recorded_turns"] for r in rows),
        "checked_turns": sum(r["checked"] for r in rows),
        "checked_scheduled": sum(r["scheduled_checked"] for r in rows),
        "max_parent_scheduled_logp_gap": max(r["max_old_logp_error"] for r in rows),
        "whole_positive": len(support),
        "whole_zero": len(rows) - len(support),
        "first_zero_turns": [(r["first_zero"] or {}).get("t") for r in rows],
        "first_zero_scheduled": sum(
            bool(r["first_zero"] and r["first_zero"]["scheduled"]) for r in rows
        ),
        "first_zero_ordinary": sum(
            bool(r["first_zero"] and not r["first_zero"]["scheduled"]) for r in rows
        ),
        "observed_episodes_with_target_outside_behavior": sum(
            r["first_target_novel"] is not None for r in rows
        ),
        "recorded_weights_ess": sum(ratios) ** 2 / sum(v * v for v in ratios)
        if any(ratios)
        else None,
        "new_original_games": 0,
    }
    save(HERE / "result.json", result)
    if a.require_recorded_overlap and (
        not support or result["observed_episodes_with_target_outside_behavior"]
    ):
        parser.exit(
            1,
            "Historical trajectories cannot provide a supported whole-episode estimate for this target.\n",
        )


if __name__ == "__main__":
    main()

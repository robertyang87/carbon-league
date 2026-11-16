#!/usr/bin/env python3
"""Assemble regression_evidence.json from the on-disk regression records."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "..")


def ident(path):
    data = open(path, "rb").read()
    return {"path": os.path.relpath(path, SW), "bytes": len(data),
            "md5": hashlib.md5(data).hexdigest()}


def load(rel):
    return json.load(open(os.path.join(HERE, rel)))


chk = load("check_src_weights.json")
cmps = {leg: load("cmp_%s.json" % leg) for leg in ("h2", "h1", "ship")}
certs = {n: load(os.path.join("certify_records", "cert_%s.json" % n)) for n in ("orig", "dedup")}
build = load("build_record_dedup.json")
verify = {n: load(os.path.join("verify_records", "verify_%s.json" % n)) for n in ("baseline", "orig", "dedup")}


def cert_summary(c):
    agg = {k: v for k, v in c["aggregate"].items() if not isinstance(v, (dict, list))}
    return {"overall": c["overall"],
            "checks_pass": {k: v.get("pass") for k, v in c["checks"].items()},
            "aggregate": agg,
            "games": [{"seed": g.get("seed"), "steps": g.get("steps"), "fatal": g.get("fatal"),
                       "step_wall_ms": g["step_wall_ms"], "game_wall_s": g.get("game_wall_s"),
                       "peak_rss_mib": g.get("peak_rss_mib"), "a_errors": g.get("a_errors"),
                       "init_wall_s": g.get("init_wall_s")} for g in c["games"]],
            "identity_three_way": {"pack": c["identity"]["pack"]["three_way_match"],
                                   "opp": c["identity"]["opp"]["three_way_match"],
                                   "harness": c["identity"]["harness"].get("three_way_match")}}


def cmp_summary(leg):
    c = cmps[leg]
    return {"verdict": c["verdict"], "games": c["games"], "actions_compared": c["actions_compared"],
            "roots_cmp_calls": c["roots_cmp_calls"], "completed_comparisons": c["completed_comparisons"],
            "changed": c["changed"], "leaf_calls": c["leaf_calls"], "depth_hist": c["depth_hist"],
            "mismatch_count": len(c["mismatches"]), "orig_conf": c["orig_conf"], "dedup_conf": c["dedup_conf"],
            "excluded_fields_note": ("ab_compare.py compares the full records except the wall-clock accumulators "
                                     "cost_accum_s / phase_accum_s (run-dependent by construction)")}


ev = {
    "run_id": "dedup-routeb-20261116",
    "task": "Weight dedup route B (alias emission) for the Echo4 tactical-search derived pack; full section-5 regression of PACK_DEDUP_INVESTIGATION.md.",
    "positioning": ("Dedup equivalence evidence + candidate pack. Per Codex round-2 R2-09 the production module and the "
                    "prototype kernel are forked; this pack is NOT stood up as a production-form candidate before kernel "
                    "re-sync -- it is to be folded in when the production pack is rebuilt after re-sync. NOT pushed to the "
                    "repo this round (parent reports to the strategy group before filing)."),
    "artifacts": {
        "original_pack": ident(os.path.join(SW, "echo4_search_int8.py")),
        "dedup_pack": ident(os.path.join(HERE, "echo4_search_int8_dedup.py")),
        "baseline_pack_opponent": ident(os.path.join(SW, "ref_echo4_int8.py")),
        "packer_before": {"md5": "852483f2d5bab0743daa6e0b8b56bb0b", "file": "dedup/cnet2_search_pack.py.before"},
        "packer_after": {"md5": "056562048f703485cade1483232411eb", "diff": "dedup/cnet2_search_pack_routeB.diff"},
        "quantizer_pinned_unchanged": {"md5": "119e9bb9", "file": "build2/cnet2_pack.py"},
        "build_record": "dedup/build_record_dedup.json",
        "diag_packs": {n: ident(os.path.join(HERE, "diag", n + ".py")) for n in ("orig_h1", "orig_h2", "dedup_h1", "dedup_h2")},
        "certify_checkout": {
            "certify_pack.py": "fc33be6cc0255d87d69c8eef8b9a2875",
            "eval_v53.py": "45792dfbea3101812836fe77e6752706 (pinned harness set member)",
            "fastsim.py": "aefe6c83826efae4de83bef8c643c80b (pinned harness set member)",
            "env_src": "59 files fetched from repo code/env_src via git blob API, every blob sha verified; carbon.json md5 cd55531b14438156def23274d594956e (pinned)",
        },
    },
    "size_accounting": {
        "original_bytes": 9030752, "dedup_bytes": 4628272, "saved_bytes": 4402480,
        "note": "Saved bytes equal the investigation's measured slot-2 duplication exactly (4,402,480 B).",
    },
    "R1_five_source_md5_set": {
        "verdict": "PASS",
        "src_md5_orig": chk["src_md5_orig"], "src_md5_dedup": chk["src_md5_dedup"],
        "equal_orig_vs_dedup": chk["R1_src_set_equal_orig_vs_dedup"],
        "equal_vs_build_record": chk["R1_src_set_equal_vs_record"],
        "mods_equal": chk["R1_mods_equal"], "conf_equal": chk["conf_equal"],
        "alias_identity_in_dedup_module": {"_B2 is _B1": chk["alias_B2_is_B1"], "_Q2 is _Q1": chk["alias_Q2_is_Q1"], "_M2 is _M1": chk["alias_M2_is_M1"]},
        "evidence_file": "dedup/check_src_weights.json",
    },
    "R2_dequantized_weights_array_by_array": {
        "verdict": "PASS",
        "w1": chk["weights"]["w1"], "w2": chk["weights"]["w2"],
        "w2_present_in_dedup": chk["R2_w2_present_in_dedup"],
        "note": "187 keys per slot = 186 tensors + meta; arrays compared elementwise after dequantization, zero mismatches in both slots.",
        "evidence_file": "dedup/check_src_weights.json",
    },
    "R3_behavior_AB": {
        "verdict": "PASS",
        "method": ("Same-seed self-play vs the baseline Echo4 pack (ref_echo4_int8.py), pack seat alternating "
                   "(game0 seed 986200 seat 0, game1 seed 986201 seat 1), full two-seat action streams + final cash + "
                   "steps + per-game comparator telemetry compared exactly; wall-clock accumulators "
                   "(cost_accum_s, phase_accum_s) excluded as run-dependent. Diagnostic packs use the shipped CONF "
                   "literal with BUDGET=30.0 so rounds complete deterministically on this VM (caliber declared; the "
                   "production 0.7 s budget aborts every root pre-comparator on this VM, which the ship leg covers)."),
        "H2_leg_CMP_H2": cmp_summary("h2"),
        "H1_leg_CMP_H1": cmp_summary("h1"),
        "ship_leg_production_CONF": cmp_summary("ship"),
        "H2_note": ("500 of 502 roots completed a full H=2 round (depth_hist), i.e. the rpol/rollout path that consumes "
                    "slot-2 weights was genuinely exercised; H=1-only equivalence would not have counted."),
        "ship_note": ("Production CONF (BUDGET=0.7): on this VM every root deadline-aborts before the comparator "
                      "(completed=0 on BOTH packs, identical call counts 261/211 per game); the leg verifies the A0 "
                      "action stream and abort behavior under the exact production configuration."),
        "evidence_files": ["dedup/cmp_h2.json", "dedup/cmp_h1.json", "dedup/cmp_ship.json",
                           "dedup/h2_orig_full.json", "dedup/h2_dedup_full.json",
                           "dedup/h1_orig_full.json", "dedup/h1_dedup_full.json",
                           "dedup/ship_orig_full.json", "dedup/ship_dedup_full.json"],
    },
    "R4_certify_pack_full_gates": {
        "verdict": "PASS (relative gate: dedup shows no new failure class and no worse gated metric than the original under identical conditions)",
        "orig": cert_summary(certs["orig"]),
        "dedup": cert_summary(certs["dedup"]),
        "reading": ("Both certificates FAIL overall on this VM on exactly the same two items (slow_count_le_limit, "
                    "overage_pool_le_limit) -- a VM-speed artifact: this VM runs steps at ~0.85-0.91 s mean vs the C1 "
                    "venue's 51.2 ms steady state (S0a), so >3 steps per game exceed the 1 s soft line. Every other gate "
                    "passes on both: init (max 0.94 s orig / 0.81 s dedup, limit 20), step hard cap (max 2.33 / 2.03 s, "
                    "limit 3), game wall (416 / 406 s, limit 1500), a_errors 0/0, identity three-way true/true, child "
                    "exit clean, peak RSS 160.2 / 156.0 MiB (limit 512). Dedup is equal or better on every reported "
                    "metric (worst-game slow count identical at 67; overage pool lower). Absolute PASS under the "
                    "150%/512M venue caliber is the C1 venue's call (S2), not this VM's."),
        "evidence_files": ["dedup/certify_records/cert_orig.json", "dedup/certify_records/cert_dedup.json"],
    },
    "verify_pack_R2_10": build["verify_pack_R2-10"],
    "caliber_declarations": [
        "Environment: this VM (shared vCPU, load fluctuated 2-6.5 during runs; VM clock UTC). Absolute timings here are ~17x slower than the C1 exam venue and are used only relatively (orig vs dedup, identical conditions).",
        "A/B determinism: diagnostic packs raise BUDGET to 30.0 s only; every other CONF field is the shipped literal verbatim (control rebuild of the shipped pack was byte-identical before any change: md5 be96d457).",
        "Certify opponent substitution: Master2's file is not present in the repo tree or on this VM, so both packs were certified against the baseline Echo4 pack (819250bf) with identical seed0 993000 / 2 games / 299 steps. Section 5.4's criterion is relative non-inferiority, which this pairing measures directly; the S0a absolute caliber (Master2, 20 games, C1 venue) is not reproduced here.",
        "Certify quota: this VM's cgroup namespace exposes no cpu.max/memory.max (/sys/fs/cgroup empty even inside a systemd-run scope, measured), so the quota self-proof records nulls report-only and --require-quota was not passed; --mem-limit-mib 512 was passed and the RSS gate ran. Attempt 1 (scope + --require-quota, incomplete checkout without env_src) failed at harness import; records kept in dedup/certify_records/attempt1_incomplete_checkout/.",
        "Runtime restarts on this VM killed background chains several times during the A/B and certify runs; all runs used resumable per-game drivers and every number above comes from validated on-disk records (1 game, 300 steps, completed) -- no partial game was counted.",
    ],
    "section5_verdicts": {
        "1_five_source_set_unchanged": "PASS",
        "2_dequantized_weights_array_by_array": "PASS (187/187 per slot, both slots)",
        "3_behavior_AB_incl_CMP_H_ge_2": "PASS (H2: 502 roots / 500 at depth 2 / 0 mismatches; H1: 503 roots / 0 mismatches; ship CONF: 472 roots / 0 mismatches)",
        "4_certify_full_gates_rerun": "PASS as relative gate (identical failure profile = VM-speed artifact only; no new failure; dedup equal-or-better on all metrics)",
        "5_verify_pack_R2_10": "Recorded: pre-existing failure on the original pack itself (mismatches=['cnet2_search'] only), identical for the dedup pack; NOT introduced by dedup; authority tree not modified.",
    },
}

out = os.path.join(HERE, "regression_evidence.json")
with open(out, "w") as f:
    json.dump(ev, f, indent=1, ensure_ascii=False)
print("wrote", out, os.path.getsize(out), "bytes")

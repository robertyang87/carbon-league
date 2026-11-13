#!/usr/bin/env python3
"""Read-only S1 gate probes: execute repository functions with synthetic rows.

No model weights, search, simulation, training, or remote access are used.
Run: python probe_method.py --repo /path/to/carbon-league --out /tmp/probe-out
"""
import argparse
import ast
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
import tempfile
import time
import types

import numpy as np


def load_functions(path):
    names = {"md5_of", "game_id_of", "sample_scene_specs", "_q_scene",
             "t_ci", "_t_crit", "bootstrap_ci", "main"}
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    constants = {"CANON", "LAYERS_Q", "LAYERS_DU", "DU_OPENING_SHARE",
                 "Q_PASS", "Q_FAIL", "DU_PASS", "ANCHOR_N"}
    body = [node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)] + body
    ns = dict(globals(), HERE=str(path.parent), _W={})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


def run_main(ns, mode, specs, qrow=None, dus=None):
    """Actual CLI main and summary; only expensive inputs/workers are replaced."""
    with tempfile.TemporaryDirectory(prefix="s1-method-probe-") as td:
        p = Path(td)
        (p / "weights.npz").write_bytes(b"fixture-no-model")
        split = {"q": ["q_fixture"], "du": ["du_fixture"]}
        (p / "split.json").write_text(json.dumps(split))
        (p / "train-ids.txt").write_text("training_game_not_in_fixture\n")
        ns["list_replays"] = lambda replay_dir, ids: list(ids)
        ns["sample_scene_specs"] = lambda *a, **k: specs
        ns["_anchor_task"] = lambda x: None
        ns["_du_scene"] = lambda x: None

        def workers(fn, tasks, count):
            if fn is ns["_q_scene"]:
                return [dict(qrow, game=s["game"], step=s["step"]) for s, _ in tasks]
            if fn is ns["_anchor_task"]:
                return [{"game": s["game"], "step": s["step"], "me": 0,
                         "du": .03, "u_q": .83, "u_g": .80} for s, _ in tasks]
            return [{"game": s["game"], "step": s["step"], "me": 0,
                     "du": float(d),
                     "u_guided": (.7501 if d >= 0 else .95) + float(d),
                     "u_greedy": .7501 if d >= 0 else .95,
                     "teacher_calls": 1, "teacher_ms": 1.0}
                    for (s, _), d in zip(tasks, dus)]

        ns["run_tasks"] = workers
        ns["T"] = types.SimpleNamespace(TEACHER_GUARD_CFG={"fixture": True})
        old_argv = sys.argv
        sys.argv = ["stage1_teacher_gate.py", "--weights", str(p / "weights.npz"),
                    "--split-manifest", str(p / "split.json"), "--replays", td,
                    "--ds-games-file", str(p / "train-ids.txt"), "--mode", mode,
                    "--out", str(p / "result.json")]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = ns["main"]()
        finally:
            sys.argv = old_argv
        result = json.loads((p / "result.json").read_text())
        return {"return_code": rc, "summary": result["q_gate" if mode == "q" else "du_gate"],
                "canonical": result["prereg"]["canonical_config"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.repo) / "code/stage1_teacher_gate.py"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ns = load_functions(src)
    sample_original = ns["sample_scene_specs"]

    # All Q are attainable expected utilities, and even attainable normal
    # terminal win utilities. Greedy arm has the largest restricted prior.
    Q = np.array([.80, .83] + [.76] * 14)
    priors = np.array([.10, .09] + [.81 / 14] * 14)
    unnorm = np.sqrt(priors) * np.exp((Q - Q.max()) / .05)
    q = unnorm / unnorm.sum()
    arms = [np.array([i], dtype=np.int64) for i in range(16)]
    teacher = {"q": q, "Q_mean": Q, "arms": arms,
               "entropy": float(-(q * np.log(q)).sum()),
               "agree_greedy": False, "wall_ms": 1.0, "guard": {}}
    ns["T"] = types.SimpleNamespace(evaluate_root=lambda *args, **kw: teacher)
    ns["extract_scene"] = lambda s: {"me": 0, "x_me": None, "aux_me": None,
        "x_opp": None, "aux_opp": None, "fm": None, "fo": None}
    ns["_fresh_sim"] = lambda s: None
    ns["_ns"] = lambda s: None
    ns["_W"]["pol"] = None
    spec = {"game": "fixture", "step": 50, "me": 0}
    cfg = {"K": 16, "M": 8, "H": 2, "seed_teacher": 7000,
           "seed_cont": 996100, "behavior_temp": 1.0}
    qrow = ns["_q_scene"]((spec, cfg))
    specs = [{"game": f"game_{i // 3:03d}", "step": [50, 150, 220][i % 3], "me": 0}
             for i in range(150)]
    q_main = run_main(ns, "q", specs, qrow=qrow)
    assert q_main["summary"]["verdict"] == "PASS"
    assert q_main["summary"]["q_soft_gain_mean"] < 0

    # Fifty source games, three positions each; roots from the same game
    # have perfectly correlated paired deltas in this lawful fixture.
    game_du = np.array([.24] * 25 + [-.16] * 25)
    root_du = np.repeat(game_du, 3)
    du_main = run_main(ns, "du", specs, dus=root_du)
    root_ci = ns["t_ci"](root_du)
    game_ci = ns["t_ci"](game_du)
    cluster_boot = ns["bootstrap_ci"](game_du, 996100)
    assert du_main["summary"]["verdict"] == "PASS"
    assert root_ci[1] > 0 and game_ci[1] < 0

    # Exercise the actual scene sampler without opening thousands of files.
    payload = json.dumps({"steps": [[{}, {}] for _ in range(300)]})
    ns["gzip"] = types.SimpleNamespace(open=lambda path: io.StringIO(payload))
    sampled = sample_original([f"/fixture/g_{i:05d}.json.gz" for i in range(10000)],
                              ns["LAYERS_DU"], 995300, opening_share=.10)
    n_open = sum(r["step"] < 40 for r in sampled)

    # N formula targets a two-sided alpha=.05 significance test against 0.
    # It does not provide 80% power for the actual conjunction with mean>=.02.
    normal = NormalDist()
    powers = []
    for sigma in [.10, .20, .30]:
        raw_n = math.ceil((140 * sigma) ** 2)
        n = min(600, max(150, raw_n))
        se = sigma / math.sqrt(n)
        sign_cutoff = normal.inv_cdf(.975) * se
        threshold = max(.02, sign_cutoff)
        powers.append({"sigma": sigma, "raw_n": raw_n, "n": n,
            "at_true_delta": .02,
            "normal_approx_significance_power": 1 - normal.cdf((sign_cutoff - .02) / se),
            "normal_approx_actual_gate_power": 1 - normal.cdf((threshold - .02) / se)})

    # Same-Q selection optimism: true action values equal .5, noisy evaluator.
    rng = np.random.default_rng(1711)
    noisy_Q = .5 + rng.uniform(-.05, .05, size=(20000, 16))
    noisy_q = np.exp((noisy_Q - noisy_Q.max(axis=1, keepdims=True)) / .05)
    noisy_q /= noisy_q.sum(axis=1, keepdims=True)
    optimistic = noisy_Q.max(axis=1) - noisy_Q[:, 0]

    evidence = {
      "source": str(src), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
      "execution_scope": "Repository _q_scene, main, t_ci, bootstrap_ci, sample_scene_specs; synthetic teacher/worker rows only; no search or training.",
      "soft_negative_pass": {"priors": priors.tolist(), "Q": Q.tolist(), "q": q.tolist(),
          "q_row": qrow, "actual_main": q_main},
      "cluster_counterexample": {"n_source_games": 50, "n_roots": 150,
          "game_deltas": game_du.tolist(), "root_t_ci": root_ci, "game_t_ci": game_ci,
          "game_boot_ci": cluster_boot, "actual_main": du_main},
      "opening_sampler": {"n_games": 10000, "n_roots": len(sampled), "n_opening": n_open,
          "observed_opening_share": n_open / len(sampled),
          "expected_opening_share": .1 / 3.1, "configured_share": .1},
      "power_normal_approximation": powers,
      "same_Q_selection_example": {"true_action_value": .5, "true_any_policy_gain": 0,
          "measured_argmax_gain": float(optimistic.mean()),
          "measured_soft_gain": float(((noisy_q * noisy_Q).sum(1) - noisy_Q[:, 0]).mean()),
          "strict_improvement_rate": float((noisy_Q.argmax(1) != 0).mean()),
          "note": "Independent terminal anchor can detect this; mean>0 sign-only check is not a 95% test."},
      "ipw_union_example": {"base_inclusion": .1, "extra_if_not_selected": .5,
          "true_inclusion": .1 + (1 - .1) * .5, "incorrect_product": .1 * .5,
          "note": "Illustrates future S1-4 inclusion-probability contract; not an implemented-code finding."}
    }
    evidence["reviewed_ref"] = "a5c63648e037e342b154949c1b4eea6d11eef139"
    evidence["findings"] = [
        {"id": "S1-M01", "severity": "P1", "kind": "implemented_gate_and_protocol_object_mismatch",
         "title": "Negative soft q gain can receive PASS", "fixture": "soft_negative_pass",
         "refs": ["code/stage1_teacher_gate.py:278", "code/stage1_teacher_gate.py:426",
                  "code/stage1_teacher_gate.py:442", "reports/STAGE1_TEACHER_GATES_20261113.md:63"]},
        {"id": "S1-M02", "severity": "P1", "kind": "statistical_independence_not_enforced",
         "title": "Multiple roots per replay treated as independent observations", "fixture": "cluster_counterexample",
         "refs": ["code/stage1_teacher_gate.py:115", "code/stage1_teacher_gate.py:220",
                  "code/stage1_teacher_gate.py:249", "code/stage1_teacher_gate.py:455"]},
        {"id": "S1-M03", "severity": "P2", "kind": "sampling_distribution_mismatch",
         "title": "10 percent games with extra root produces about 3.2 percent opening roots", "fixture": "opening_sampler",
         "refs": ["code/stage1_teacher_gate.py:84", "code/stage1_teacher_gate.py:125"]},
        {"id": "S1-M04", "severity": "P2", "kind": "power_reporting_and_protocol_clarity",
         "title": "Capped sample size omits promised power; formula does not power full point-threshold conjunction",
         "fixture": "power_normal_approximation",
         "refs": ["reports/STAGE1_TEACHER_GATES_20261113.md:82", "code/stage1_teacher_gate.py:465"]},
    ]
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"q_verdict": q_main["summary"]["verdict"],
                      "soft_gain": q_main["summary"]["q_soft_gain_mean"],
                      "du_verdict": du_main["summary"]["verdict"],
                      "root_ci": root_ci, "cluster_ci": game_ci,
                      "opening_share": n_open / len(sampled),
                      "evidence": str(out / "evidence.json")}, indent=2))


if __name__ == "__main__":
    main()

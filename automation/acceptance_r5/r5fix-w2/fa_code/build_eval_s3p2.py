"""Generate eval_s3p2.py / record_s3p2_traces.py from the S3 v1 scripts
with targeted replacements (files, bot s3p2 stats, job sets, results
file, INDEX section)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

t = open(os.path.join(HERE, "eval_s3.py")).read()

subs = [
    ('"""eval_s3: paired-seed ablation for the S3 structural line\n'
     '(fork_s3_phase: P6 sprint phase + P5 hunt feedback loop) vs base_v53.',
     '"""eval_s3p2: paired-seed ablation for the S3 v2 structural line\n'
     '(fork_s3_phase2: P6 sprint phase recalibrated -- ratio modes +\n'
     'hysteresis + banking + protect fade + widened recruit gate;\n'
     'P5 hunt loop byte-identical to v1) vs base_v53.'),
    ('BOTH = "fork_s3_phase.py"', 'BOTH = "fork_s3_phase2.py"'),
    ('PONLY = "fork_s3_ponly.py"', 'NOREC = "fork_s3p2_norec.py"'),
    ('HONLY = "fork_s3_honly.py"', 'HONLY = "fork_s3p2_honly.py"'),
    ('OFF = "fork_s3_off.py"', 'OFF = "fork_s3p2_off.py"'),
    ('''    st = getattr(bot, "s3_stats", None)
    if st is not None:
        out["s3"] = {k: v for k, v in st.items()}
        out["s3"]["hunt_level_final"] = getattr(bot, "s3_hunt_level", None)
    return out''',
     '''    st = getattr(bot, "s3_stats", None)
    if st is not None:
        out["s3"] = {k: v for k, v in st.items()}
        out["s3"]["hunt_level_final"] = getattr(bot, "s3_hunt_level", None)
    st2 = getattr(bot, "s3p2_stats", None)
    if st2 is not None:
        out["s3p2"] = {k: v for k, v in st2.items()}
        out["s3p2"]["mode_final"] = getattr(bot, "s3p2_mode", None)
    return out'''),
    ('''  smoke    both-on vs base, seed 999 x2 (run with CARBON_DEBUG=1)
  sanity   off-variant vs base + base vs base, seed 1000 x2
           (off must reproduce base-vs-base margins exactly)
  main     both-on seeds 1000-1003 x2 (8), phase-only seeds
           1000-1001 x2 (4), hunt-only seeds 1000-1001 x2 (4)
Results append to eval_s3_results.jsonl.''',
     '''  smoke    both2 vs base, seed 999 x2
  sanity   off2-variant vs base + base vs base, seed 1000 x2
           (off2 must reproduce base-vs-base margins exactly;
           honly2 must reproduce v1 s3honly margins exactly)
  main     both2 seeds 1000-1003 x2 (8), honly2 seeds
           1000-1003 x2 (8) -- phase net effect decomposable on
           all 8 paired games
  abl      norec (v1 recruit bars) seeds 1000-1001 x2 (4)
Results append to eval_s3p2_results.jsonl.'''),
    ('''    if phase == "smoke":
        return [("s3smoke", BOTH, BASE, 999, side) for side in (0, 1)]
    if phase == "sanity":
        return ([("s3off", OFF, BASE, 1000, side) for side in (0, 1)]
                + [("basebase", BASE, BASE, 1000, side) for side in (0, 1)])
    if phase == "ctrl":
        # supplementary control: hunt-only on the both-on-only seeds,
        # so the phase effect is decomposable on all 8 paired games
        return [("s3honly", HONLY, BASE, s, side)
                for s in (1002, 1003) for side in (0, 1)]
    if phase == "main":
        return ([("s3both", BOTH, BASE, s, side)
                 for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
                + [("s3ponly", PONLY, BASE, s, side)
                   for s in (1000, 1001) for side in (0, 1)]
                + [("s3honly", HONLY, BASE, s, side)
                   for s in (1000, 1001) for side in (0, 1)])''',
     '''    if phase == "smoke":
        return [("s3p2smoke", BOTH, BASE, 999, side) for side in (0, 1)]
    if phase == "sanity":
        return ([("s3p2off", OFF, BASE, 1000, side) for side in (0, 1)]
                + [("basebase", BASE, BASE, 1000, side) for side in (0, 1)])
    if phase == "main":
        return ([("s3p2both", BOTH, BASE, s, side)
                 for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
                + [("s3p2honly", HONLY, BASE, s, side)
                   for s in (1000, 1001, 1002, 1003) for side in (0, 1)])
    if phase == "abl":
        return [("s3p2norec", NOREC, BASE, s, side)
                for s in (1000, 1001) for side in (0, 1)]'''),
    ('open(os.path.join(HERE, "eval_s3_results.jsonl"), "a")',
     'open(os.path.join(HERE, "eval_s3p2_results.jsonl"), "a")'),
]
for a, b in subs:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open(os.path.join(HERE, "eval_s3p2.py"), "w").write(t)

r = open(os.path.join(HERE, "record_s3_traces.py")).read()
rsubs = [
    ("record_s3_traces: record all eval_s3 games",
     "record_s3p2_traces: record all eval_s3p2 games"),
    ("eval_s3_results.jsonl (exact 0.00 required), and append the S3 section",
     "eval_s3p2_results.jsonl (exact 0.00 required), and append the S3 v2 section"),
    ("import eval_s3\nimport eval_v53", "import eval_s3p2 as eval_s3\nimport eval_v53"),
    ('(eval_s3.jobs_for("main")\n                                          + eval_s3.jobs_for("ctrl")\n                                          + eval_s3.jobs_for("sanity"))',
     '(eval_s3.jobs_for("main")\n                                          + eval_s3.jobs_for("abl")\n                                          + eval_s3.jobs_for("sanity"))'),
    ('"eval_s3_results.jsonl"', '"eval_s3p2_results.jsonl"'),
    ('''    names = {
        "fork_s3_phase": "fork-s3 全开",
        "fork_s3_ponly": "fork-s3 仅相位",
        "fork_s3_honly": "fork-s3 仅猎杀",
        "fork_s3_off": "fork-s3 双关(sanity)",
        "base_v53": "base_v53（基座）",
    }''',
     '''    names = {
        "fork_s3_phase2": "fork-s3p2 全开",
        "fork_s3p2_honly": "fork-s3p2 仅猎杀",
        "fork_s3p2_off": "fork-s3p2 双关(sanity)",
        "fork_s3p2_norec": "fork-s3p2 相位+原补员闸(消融)",
        "base_v53": "base_v53（基座）",
    }'''),
    ('"## S3 终局相位＋猎杀反馈闭环（eval_s3，2026-11-03）"',
     '"## S3 二版：终局相位重标定（eval_s3p2，2026-11-03）"'),
    ('print("INDEX.md S3 section appended.")',
     'print("INDEX.md S3 v2 section appended.")'),
]
for a, b in rsubs:
    assert r.count(a) == 1, a[:60]
    r = r.replace(a, b)
open(os.path.join(HERE, "record_s3p2_traces.py"), "w").write(r)
print("generated eval_s3p2.py / record_s3p2_traces.py")

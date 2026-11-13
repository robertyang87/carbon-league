"""analyze_s1: aggregate eval_s1_results.jsonl into the numbers quoted
in s1_treeval_report.md (稳/准/狠 scorecard + ablation table + ledger
comparison). Prints JSON to stdout."""

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
with open(os.path.join(HERE, "eval_s1_results.jsonl")) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

tags = {}
for r in rows:
    tags.setdefault(r["tag"], []).append(r)

out = {}
for tag, rs in sorted(tags.items()):
    margins = [r["margin"] for r in rs]
    per_seed = {}
    for r in rs:
        per_seed[r["seed"]] = per_seed.get(r["seed"], 0.0) + r["margin"]
    led_a = {}
    led_b = {}
    for k in ("deposits", "tree_income", "plant_count", "seize_count",
              "plant_spend", "seize_spend", "deaths", "death_cargo",
              "stranded"):
        led_a[k] = round(st.mean(r["ledger_pol"][k] for r in rs), 1)
        led_b[k] = round(st.mean(r["ledger_opp"][k] for r in rs), 1)
    trees100_a = round(st.mean(r["comp"]["100"][r["policy_side"]][2]
                               for r in rs), 2)
    trees100_b = round(st.mean(r["comp"]["100"][1 - r["policy_side"]][2]
                               for r in rs), 2)
    # 狠 aggregates (pol = variant side, opp = base side)
    def agg(which, field):
        return [r[which][field] for r in rs]
    ruth_a = {
        "spells": round(st.mean(agg("ruth_pol", "spells")), 1),
        "seize_rate": round(
            sum(r["ruth_pol"]["seized_away"] for r in rs) /
            max(1, sum(r["ruth_pol"]["seized_away"] + r["ruth_pol"]["died"]
                       for r in rs)), 3),
        "seize_flips": round(st.mean(agg("ruth_pol", "seize_flips")), 1),
        "recap_n": sum(len(r["ruth_pol"]["recap_delays"]) for r in rs),
        "recap_med": (st.median([d for r in rs
                                 for d in r["ruth_pol"]["recap_delays"]])
                      if any(r["ruth_pol"]["recap_delays"] for r in rs)
                      else None),
        "planted": round(st.mean(agg("ruth_pol", "planted_trees")), 1),
        "payback_rate": round(
            sum(r["ruth_pol"]["payback_n"] for r in rs) /
            max(1, sum(r["ruth_pol"]["planted_trees"] for r in rs)), 3),
        "payback_med": (st.median([d for r in rs
                                   for d in r["ruth_pol"]["payback_delays"]])
                        if any(r["ruth_pol"]["payback_delays"] for r in rs)
                        else None),
    }
    ruth_b = {
        "spells": round(st.mean(agg("ruth_opp", "spells")), 1),
        "seize_rate": round(
            sum(r["ruth_opp"]["seized_away"] for r in rs) /
            max(1, sum(r["ruth_opp"]["seized_away"] + r["ruth_opp"]["died"]
                       for r in rs)), 3),
        "seize_flips": round(st.mean(agg("ruth_opp", "seize_flips")), 1),
        "recap_n": sum(len(r["ruth_opp"]["recap_delays"]) for r in rs),
        "recap_med": (st.median([d for r in rs
                                 for d in r["ruth_opp"]["recap_delays"]])
                      if any(r["ruth_opp"]["recap_delays"] for r in rs)
                      else None),
        "planted": round(st.mean(agg("ruth_opp", "planted_trees")), 1),
        "payback_rate": round(
            sum(r["ruth_opp"]["payback_n"] for r in rs) /
            max(1, sum(r["ruth_opp"]["planted_trees"] for r in rs)), 3),
        "payback_med": (st.median([d for r in rs
                                   for d in r["ruth_opp"]["payback_delays"]])
                        if any(r["ruth_opp"]["payback_delays"] for r in rs)
                        else None),
    }
    # 准: prediction calibration from matched preds (non-censored only)
    preds = [p for r in rs for p in r["preds"] if p["matched"]]
    cal = {}
    for kind in ("plant", "seize", "all"):
        sub = [p for p in preds
               if (kind == "all" or p["kind"] == kind)
               and p["how"] != "censored"]
        if not sub:
            cal[kind] = None
            continue
        cal[kind] = {
            "n": len(sub),
            "n_censored": sum(1 for p in preds
                              if (kind == "all" or p["kind"] == kind)
                              and p["how"] == "censored"),
            "mae_new": round(st.mean(abs(p["pin_new"] - p["realized"])
                                     for p in sub), 1),
            "mae_old": round(st.mean(abs(p["pin_old"] - p["realized"])
                                     for p in sub), 1),
            "bias_new": round(st.mean(p["pin_new"] - p["realized"]
                                      for p in sub), 1),
            "bias_old": round(st.mean(p["pin_old"] - p["realized"]
                                      for p in sub), 1),
            "mean_real": round(st.mean(p["realized"] for p in sub), 1),
            "seized_frac": round(
                sum(1 for p in sub if p["how"] == "seized_away") / len(sub), 3),
        }
    out[tag] = {
        "n": len(rs),
        "wins": sum(1 for r in rs if r["win"]),
        "mean_margin": round(st.mean(margins), 1),
        "margins": margins,
        "per_seed_pair": {str(k): round(v, 1)
                          for k, v in sorted(per_seed.items())},
        "worst_game": round(min(margins), 1),
        "worst_seed_pair": round(min(per_seed.values()), 1),
        "a_ms": round(st.mean(r["a_ms_per_step"] for r in rs), 1),
        "b_ms": round(st.mean(r["b_ms_per_step"] for r in rs), 1),
        "ledger_a": led_a, "ledger_b": led_b,
        "trees100_a": trees100_a, "trees100_b": trees100_b,
        "ruth_a": ruth_a, "ruth_b": ruth_b,
        "cal": cal,
        "preds_matched": len(preds),
        "preds_total": sum(len(r["preds"]) for r in rs),
    }

print(json.dumps(out, ensure_ascii=False, indent=1))

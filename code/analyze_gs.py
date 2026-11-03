"""analyze_gs: paired-delta + trigger audit for the GS line.

Reads eval_gs_results.jsonl (disk only) and the S2 control rows in
eval_s2_results.jsonl:
  * chk_harvest rows (switcher_harvest as policy vs base) give the
    base-vs-harvest baseline in the mirrored slot (base player =
    1 - policy_side, base margin = -margin, base ledger = ledger_opp)
  * gs_base_h2 rows in eval_gs_results.jsonl are the base-vs-harvest
    control for the expansion seeds 1002-1003
  * ctrl_base rows (fork_s2_off vs base, decision-identical to base)
    give the mirror control for the variant-vs-base side-effect games

Trigger audit: reconstructs the trigger step from the recorded
enemy-planter trajectory (enp_hist, pre-step counts, harness step =
index + 1) under both the armed rule used by the variants and the
raw unarmed rule, and compares with the bot's own gs_trigger_step
(bot obs-step units; |diff| <= 1 expected from the pre/post offset).

Writes gs_analysis.json and prints the tables used in the report.
"""

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    rows = []
    p = os.path.join(HERE, fn)
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def recon_trigger(enp_hist, armed):
    """First harness step (1-based) at which the consecutive-zero run
    of enemy planter counts reaches 12; None if never."""
    is_armed = not armed
    run = 0
    for i, c in enumerate(enp_hist):
        if c > 0:
            is_armed = True
            run = 0
        elif is_armed:
            run += 1
            if run >= 12:
                return i + 1
    return None


def max_run_after_arming(enp_hist):
    armed = False
    run = 0
    best = 0
    for c in enp_hist:
        if c > 0:
            armed = True
            run = 0
        elif armed:
            run += 1
            best = max(best, run)
    return best


def main():
    gs = load("eval_gs_results.jsonl")
    s2 = load("eval_s2_results.jsonl")

    # ---- baseline slots: base vs switcher_harvest ------------------
    # (seed, base_player) -> (base margin, base ledger, base cash,
    #                         harvest cash)
    base_slot = {}
    for r in s2:
        if r["tag"] == "chk_harvest":
            base_slot[(r["seed"], 1 - r["policy_side"])] = (
                -r["margin"], r["ledger_opp"], r["opp_cash"],
                r["my_cash"])
    for r in gs:
        if r["tag"] == "gs_base_h2":
            base_slot[(r["seed"], r["policy_side"])] = (
                r["margin"], r["ledger_pol"], r["my_cash"],
                r["opp_cash"])
    # same-convention control (this harness, base as policy) overrides
    # the cross-convention chk rows wherever it exists
    for r in gs:
        if r["tag"] == "gs_base_h1":
            base_slot[(r["seed"], r["policy_side"])] = (
                r["margin"], r["ledger_pol"], r["my_cash"],
                r["opp_cash"])

    # ---- mirror control: variant vs base ----------------------------
    ctrl_slot = {}   # (seed, player) -> margin of decision-identical off
    for r in s2:
        if r["tag"] == "ctrl_base":
            ctrl_slot[(r["seed"], r["policy_side"])] = r["margin"]

    out = {"base_slot_margins": {f"{k[0]}/{k[1]}": v[0]
                                 for k, v in sorted(base_slot.items())},
           "variants": {}, "side_effects": {}, "triggers": []}

    # ---- variant vs harvest ----------------------------------------
    for tag in ("gs_seize_h", "gs_seize_h2", "gs_guard_h",
                "gs_guard_h2", "gs_both_h"):
        rows = [r for r in gs if r["tag"] == tag]
        if not rows:
            continue
        slots = []
        for r in rows:
            key = (r["seed"], r["policy_side"])
            bm, bled, bcash, hcash = base_slot[key]
            L, BL = r["ledger_pol"], bled
            slots.append(dict(
                seed=r["seed"], side=r["policy_side"],
                margin=r["margin"], base_margin=bm,
                delta=round(r["margin"] - bm, 2),
                exact_base=(r["my_cash"] == bcash
                            and r["opp_cash"] == hcash),
                seize=L["seize_count"], seize_base=BL["seize_count"],
                seized_inc=L["seized_income"],
                seized_inc_base=BL.get("seized_income"),
                tree_inc=L["tree_income"],
                tree_inc_base=BL["tree_income"],
                deposits=L["deposits"], deposits_base=BL["deposits"],
                plants=L["plant_count"], plants_base=BL["plant_count"],
                plant_spend=L["plant_spend"],
                plant_spend_base=BL["plant_spend"],
                seize_spend=L["seize_spend"],
                trigger=(r["gs_pol"] or {}).get("trigger_step")))
        ds = [s["delta"] for s in slots]
        out["variants"][tag] = dict(
            n=len(slots), mean_delta=round(sum(ds) / len(ds), 1),
            worst_delta=min(ds), slots=slots)
        print(f"{tag}: n={len(slots)} meanΔ={sum(ds)/len(ds):+.1f} "
              f"worstΔ={min(ds):+.0f} deltas={[s['delta'] for s in slots]}")

    # ---- variant vs base (side effects) -----------------------------
    for tag in ("gs_seize_base", "gs_guard_base"):
        rows = [r for r in gs if r["tag"] == tag]
        if not rows:
            continue
        slots = []
        for r in rows:
            key = (r["seed"], r["policy_side"])
            cm = ctrl_slot.get(key)
            slots.append(dict(
                seed=r["seed"], side=r["policy_side"],
                margin=r["margin"], ctrl_margin=cm,
                delta_vs_ctrl=(round(r["margin"] - cm, 2)
                               if cm is not None else None),
                trigger=(r["gs_pol"] or {}).get("trigger_step"),
                max_run=(r["gs_pol"] or {}).get("max_run")))
        out["side_effects"][tag] = slots
        print(f"{tag}: margins={[s['margin'] for s in slots]} "
              f"triggers={[s['trigger'] for s in slots]}")

    # ---- trigger audit (all gs games) --------------------------------
    for r in gs:
        if r["gs_pol"] is None and r["tag"] == "gs_base_h2":
            bot_trig = None
        else:
            bot_trig = (r["gs_pol"] or {}).get("trigger_step")
        rec_armed = recon_trigger(r["enp_hist"], armed=True)
        rec_raw = recon_trigger(r["enp_hist"], armed=False)
        first_p = next((i + 1 for i, c in enumerate(r["enp_hist"])
                        if c > 0), None)
        out["triggers"].append(dict(
            tag=r["tag"], seed=r["seed"], side=r["policy_side"],
            opp=r["b_file"], bot_trigger=bot_trig,
            recon_armed=rec_armed, recon_raw=rec_raw,
            first_enemy_planter_step=first_p,
            max_run_enp0=max_run_after_arming(r["enp_hist"])))
    for t in out["triggers"]:
        print(f"trig {t['tag']} s{t['seed']}/{t['side']} vs {t['opp']}: "
              f"bot={t['bot_trigger']} armed={t['recon_armed']} "
              f"raw={t['recon_raw']} firstP={t['first_enemy_planter_step']} "
              f"maxrun={t['max_run_enp0']}")

    with open(os.path.join(HERE, "gs_analysis.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote gs_analysis.json")


if __name__ == "__main__":
    main()

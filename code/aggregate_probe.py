"""aggregate_probe: summarise eval_probe measurement output.

Reads eval_probe_results.jsonl + probe_stats/*.json and prints, per
opponent and per game: headline avoidance tax and its decomposition,
enemy strike rates by cargo bucket, and attributed interception losses.
Output is the raw material for avoid_tax_report.md (numbers only from
disk, per house discipline).
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPP_NAME = {"pbshadow": "shadow_alpha_v2", "pbv05": "policy_v0_5",
            "pbex": "example"}
BUCKET_ORDER = ["0", "1-80", "81-150", "151-250", "251+"]


def main():
    rows = []
    with open(os.path.join(HERE, "eval_probe_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["tag"], r["policy_side"]))
    by_opp = {}
    for r in rows:
        sp = r.get("stats_path")
        stats = None
        if sp and os.path.exists(sp):
            with open(sp) as f:
                stats = json.load(f)
        r["_stats"] = stats
        by_opp.setdefault(r["tag"], []).append(r)

    for tag, games in by_opp.items():
        print(f"\n===== vs {OPP_NAME[tag]} ({tag}) =====")
        for r in games:
            S = r["_stats"]
            print(f"-- seed{r['seed']} side{r['policy_side']}: "
                  f"cash {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
                  f"(margin {r['margin']:+.2f}), "
                  f"probe {r['a_ms_per_step']} ms/step")
            if S is None:
                print("   STATS FILE MISSING")
                continue
            tax_dp = S["regret_all_free"]
            tax_dp_mr = S["regret_mr_free"]
            tax_let = S["let_loss"]
            headline = tax_dp + tax_let
            print(f"   TAX headline={headline:.1f} "
                  f"[DP regret_all(free)={tax_dp:.1f} "
                  f"(minreach-only {tax_dp_mr:.1f}, soft marginal "
                  f"{tax_dp - tax_dp_mr:.1f}; role regret_all="
                  f"{S['regret_all_role']:.1f}; final-action regret "
                  f"cross-check={S['regret_final']:.1f} over "
                  f"{S['final_turns']} turns) + lethal let_loss="
                  f"{tax_let:.1f} (lethal+tie+eg all_loss={S['all_loss']:.1f})]")
            print(f"   DP turns free={S['dp_turns_free']} role={S['dp_turns_role']}; "
                  f"plan changed by penalties: all {S['chg_all_free']}x, "
                  f"minreach-only {S['chg_mr_free']}x; burden(free)="
                  f"{S['burden_free']:.1f}")
            print(f"   lethal: turns charged={S['let_turns']}, "
                  f"charged actions={S['let_charged_actions']}, "
                  f"reroutes={S['let_reroute']}; all-layer reroutes="
                  f"{S['all_reroute']}")
            op = S["opp_pairs"]
            print(f"   strike: enemy turns={S['en_turns']} "
                  f"(gone {S['en_gone']}); opp pairs={op}, strikes="
                  f"{S['strike_pairs']} "
                  f"({100.0 * S['strike_pairs'] / op:.1f}% of opps)"
                  f", approaches={S['approach_pairs']} "
                  f"({100.0 * S['approach_pairs'] / op:.1f}%)"
                  if op else "   strike: no opportunities")
            print(f"   strike turn-level: opp turns={S['opp_turns']}, "
                  f"strike turns={S['strike_turns']}, approach turns="
                  f"{S['approach_turns']}; tie opps={S['tie_opp_pairs']}, "
                  f"tie strikes={S['tie_strike_pairs']}")
            print(f"   losses: clean intercept {S['loss_clean']:.1f} "
                  f"x{S['loss_clean_n']}, tie {S['loss_tie']:.1f} "
                  f"x{S['loss_tie_n']}, own-collision {S['loss_own']:.1f} "
                  f"x{S['loss_own_n']}, other {S['loss_other']:.1f} "
                  f"x{S['loss_other_n']}, unknown {S['loss_unknown']:.1f} "
                  f"x{S['loss_unknown_n']}, enemy-base {S['loss_enbase']:.1f} "
                  f"x{S['loss_enbase_n']}; my collector deaths="
                  f"{S['deaths_c']}, my collector turns={S['my_c_turns']}, "
                  f"exposure(>=100)={S['exposure100']}")
            print(f"   ledger death_cargo (env truth): mine="
                  f"{r['ledger_pol']['death_cargo']:.1f}, opp="
                  f"{r['ledger_opp']['death_cargo']:.1f}; my deaths="
                  f"{r['ledger_pol']['deaths']}, opp deaths="
                  f"{r['ledger_opp']['deaths']}")
        # opponent-level means
        taxes = []
        for r in games:
            S = r["_stats"]
            if S:
                taxes.append(S["regret_all_free"] + S["let_loss"])
        if taxes:
            print(f"   >> mean headline tax vs {OPP_NAME[tag]}: "
                  f"{sum(taxes) / len(taxes):.1f}/game "
                  f"(games: {', '.join(f'{t:.1f}' for t in taxes)})")
        # merged buckets across the pair
        tot = {}
        for r in games:
            S = r["_stats"]
            if not S:
                continue
            for b, (o, s, a) in S["buckets"].items():
                t = tot.setdefault(b, [0, 0, 0])
                t[0] += o
                t[1] += s
                t[2] += a
        if tot:
            print("   >> strike buckets (enemy cargo -> opp/strike/approach):")
            for b in BUCKET_ORDER:
                if b in tot:
                    o, s, a = tot[b]
                    print(f"      {b:>7}: opp={o} strike={s} "
                          f"({100.0 * s / o:.1f}%) approach={a} "
                          f"({100.0 * a / o:.1f}%)" if o else
                          f"      {b:>7}: opp=0")


if __name__ == "__main__":
    main()

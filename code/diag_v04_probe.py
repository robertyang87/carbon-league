"""diag_v04_probe: run ONE game with a policy file and print the
ledger + v0.4 tree-warfare probe counters as a table.

  python3 diag_v04_probe.py POLICY_FILE SEED SIDE [OPP]

Default OPP=reference_juzi. Used for the v0.4 diagnose-first step:
the instrumented build (plant_home_on=0, probe read-only) must
reproduce v0.3's recorded score for the same seed/side exactly.
"""

import json
import sys

import eval_v0_4


def main():
    policy_file = sys.argv[1]
    seed = int(sys.argv[2])
    side = int(sys.argv[3])
    opp = sys.argv[4] if len(sys.argv) > 4 else "reference_juzi"
    res = eval_v0_4.play_game((seed, side, opp, policy_file, "diag_v04"))
    print(f"RESULT {res['my_cash']:.2f} : {res['opp_cash']:.2f} "
          f"margin {res['margin']:.2f}  ms/step {res['my_ms_per_step']}")
    print("ledger_pol:", json.dumps(res["ledger_pol"], sort_keys=True))
    print("ledger_opp:", json.dumps(res["ledger_opp"], sort_keys=True))
    tw = res.get("tw") or {}
    detail = tw.pop("tw_targets_detail", None)
    print("--- tree-warfare probe counters ---")
    for k in sorted(tw):
        print(f"  {k:28s} {tw[k]}")
    if detail:
        print("  targets still tracked at end:", detail)


if __name__ == "__main__":
    main()

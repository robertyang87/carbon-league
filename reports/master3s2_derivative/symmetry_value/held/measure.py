"""Run the original real-clock cross-over on independently registered held maps; select at most one follow-up."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from search_check import main as measure_panel
from search_settle import settlement
PROTOCOL_SHA="4491eb8d723a2bec1f3cf12389a3d273a9f86fed95844a8304acf53d22d3c9e1"


def run():
    dest=HERE/"search_roots.json.gz"
    measure_panel(HERE/"protocol.json",PROTOCOL_SHA,"val",dest)
    summary=settlement(dest)
    rows=summary["roots"]
    admissible=[r for r in rows if r["attributable_changed"] and r["all_original_and_new_abs_gt_1e_5"]
                and r["first_trial_physics"] and all(not v["same_official_next"] for v in r["first_trial_physics"])]
    first=admissible[0] if admissible and summary["protocol_gate"] else None
    summary["first_eligible_root"]={k:first[k] for k in ("seed","seat","step")} if first else None
    with (HERE/"search_settlement.json").open("x") as f:
        json.dump(summary,f,sort_keys=True,allow_nan=False,indent=2)
        f.write("\n")
    print(summary["first_eligible_root"],flush=True)


if __name__=="__main__":
    run()

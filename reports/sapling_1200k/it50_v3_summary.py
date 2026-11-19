"""Summarize the two legacy records and the four fresh games without dropping losses."""
from collections import Counter
import json
from pathlib import Path
import statistics

REPORTS = Path(__file__).resolve().parent


def outcome(record, *, legacy=False):
    if legacy:
        r = record
        side = r["candidate_seat"]
        moments = {int(t): [s["candidate"], s["opponent"]] for t,s in r["snapshots"].items()}
        attrs = r["action_attribution_on_shared_prefix"]
        stats = r["search"]
        incomes = [r["final"][role]["cumulative_tree_absorption_income"] for role in ("candidate","opponent")]
        maps = r["seed"]
        audit = not any(r["agent_errors_by_seat"])
        calls = r["agent_calls_by_seat"][1-side]
    else:
        r = record
        side = r["seat"]
        moments = {int(t): [s[side],s[1-side]] for t,s in r["snapshots_by_step_and_seat"].items()}
        attrs = r["own_model_action_attribution"]
        stats = r["raw_search_stats"]
        incomes = [r["cumulative_tree_income_by_seat"][s] for s in (side,1-side)]
        maps = r["map"]
        audit = r.get("audit_passed",True) and not any(r["agent_errors_by_seat"])
        calls = r["source_search_invocations"]
    if not audit:
        raise ValueError(f"keep failed audit out of certified sample; consult raw {maps}/{side}")
    terminal = r["actual_transitions"]
    last = moments[terminal]
    return {
        "map": maps, "seat": side, "result": r["result"], "margin": r["margin"],
        "official_scores_candidate_first": [r["official_score_by_seat"][i] for i in (side,1-side)],
        "t": terminal, "search_calls": calls, "depth8": stats["depth"].get("8", 0), "aborted": stats["aborted"],
        "candidate_wall_ms": 1000*r["call_seconds_by_seat"][side]["mean"],
        "examiner_wall_ms": 1000*r["call_seconds_by_seat"][1-side]["mean"],
        "cash_margins": {str(t):round(s[0]["cash"]-s[1]["cash"],2) for t,s in moments.items()},
        "deposited": [round(s["own_positive_delivery_carbon"],2) for s in last],
        "tree_income": [round(v,2) for v in incomes], "new_trees": [s["new_trees_planted"] for s in last],
        "ending_cargo": [round(s["cargo"],2) for s in last],
        "joint_prefix_disagreements": sum(v["disagree"] for v in attrs.values()),
        "student_choices_in_disagreements": sum(v["selected_student_in_disagreement"] for v in attrs.values()),
        "third_choices_in_disagreements": sum(v["selected_third_in_disagreement"] for v in attrs.values()),
    }


def summary(rows):
    return {"maps": len({r["map"] for r in rows}), "n": len(rows), "win_draw_loss": dict(Counter(r["result"] for r in rows)),
            "mean_margin": statistics.mean(r["margin"] for r in rows),
            "full_299": sum(r["t"] == 299 for r in rows)}


def main():
    old = json.loads((REPORTS/"it50_v3_exam.json").read_text())
    original = list(map(lambda r: outcome(r,legacy=True),old["games"]))
    original.append(outcome(json.loads((REPORTS/"it50_v3_seat1_repeat.json").read_text())))
    folder = REPORTS/"it50_v3_fresh_games"
    block = json.loads((folder/"index.json").read_text())
    if block["status"] != "complete":
        raise RuntimeError(f"fresh block incomplete: {block['status']}")
    fresh = [outcome(json.loads((folder/g["artifact"]).read_text())) for g in block["games"]]
    assert [(r["map"],r["seat"]) for r in fresh] == [(s,i) for s in (620098121,620098122) for i in (0,1)]
    all_games = original + fresh
    print(json.dumps({"original":summary(original),"fresh":summary(fresh),"combined":summary(all_games),
                      "unscored_original_seat1_attempt":1,"scored_games":all_games}, indent=2))


if __name__ == "__main__":
    main()

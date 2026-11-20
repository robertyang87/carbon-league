"""Check each source/branch handoff and replay the entire recorded history in the official referee."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import select_roots as sel

common=sel.common
HERE=sel.HERE
spec=importlib.util.spec_from_file_location("conditional_trajectory_verifier",common.HERE/"verify.py")
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipts",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    d=args.out
    summary=audit.load(d/"summary.json")
    tickets=audit.load(HERE/"roots.json")["tickets"]
    ticket=next(t for t in tickets if t["seed"]==summary["seed"])
    seat=ticket["seat"]; start=ticket["root_step"]
    source_path=args.receipts/"fit"/f'{ticket["seed"]}_{seat}'/"source.json.gz"
    common.need(common.sha(source_path.read_bytes())==ticket["source_sha256"]==summary["source_sha256"],"Bad source")
    source=audit.load(source_path)
    root=source["rows"][start]
    protocol=audit.load(HERE/"protocol.json")
    common.need(summary["protocol_sha256"]==common.sha((HERE/"protocol.json").read_bytes()) and
                protocol["checker_sha256"]==common.sha(Path(__file__).read_bytes()),"Plan/checker mismatch")
    common.need(common.sha((d/"prefix.json.gz").read_bytes())==summary["prefix_sha256"],"Warm prefix changed")
    warm=audit.load(d/"prefix.json.gz")
    for i,r in enumerate(warm["rows"]):
        original=source["rows"][i]
        common.need(r["native_public_sha256"]==original["enemy_public_sha256"] and
                    r["historical_order"]==original["orders"][1-seat] and
                    r["historical"]==original["clocks"][1-seat] and r["restored_native_cost"]==r["historical"]["cost"],
                    "History replay drift")
    common.need(len(warm["rows"])==start+1 and warm["root_native_cost"]==root["clocks"][1-seat]["cost"],
                "Wrong native clock initialization")
    terminal={}
    arm_steps={}
    divergence=None
    for name,record in summary["arms"].items():
        path=d/(name+".json.gz")
        common.need(common.sha(path.read_bytes())==record["sha256"],"Arm changed")
        arm=audit.load(path);rows=arm["rows"]
        common.need(arm["root_native_counterfactual_sha256"]==summary["root_native_counterfactual_sha256"]
                    and arm["root_public_sha256"]==ticket["root_public_sha256"]==common.digest(root["public"])
                    and arm["root_board_sha256"]==root["board_before"],"Wrong paired root")
        common.need(rows[0]["orders"][seat]==ticket["orders"][name] and rows[0]["orders"][1-seat]==root["orders"][1-seat]
                    and rows[0]["clocks"][1-seat]==root["clocks"][1-seat],"Wrong shared or student first action")
        for j,row in enumerate(rows):
            common.need(row["active_policy"]==(name if j<8 else "bc") and row["shared_historical_opponent_command"]==(j==0),
                        "Wrong mixed policy window")
        trajectory=source["rows"][:start]+rows
        audit.replay(source,trajectory,arm["terminal"])
        common.need(arm["inprocess_time_limits"]==common.local_time_limits(trajectory),"Diagnostic resource accounting")
        terminal[name]=arm["terminal"]
        arm_steps[name]=rows
    common.need(set(terminal)=={"weighted","uniform"} and
                summary["weighted_minus_uniform"]==terminal["weighted"]["margin"]-terminal["uniform"]["margin"]
                and summary["win_delta"]==terminal["weighted"]["win"]-terminal["uniform"]["win"],"Missing or wrong score")
    common.need(arm_steps["weighted"][0]["orders"][seat]!=arm_steps["uniform"][0]["orders"][seat],"No effective intervention")
    out={"seed":ticket["seed"],"seat":seat,"root":start,"two_root_actions_confirmed":True,"official_terminals":terminal,
         "relative_official_margin":summary["weighted_minus_uniform"],"native_rng_in_historical_source_available":False,
         "restored_historical_cost_vectors":len(warm["rows"]),"prefix_disagreement_repaired":sum(r["corrected"] for r in warm["rows"]),
         "native_rng_and_cost_fingerprint_for_both_suffixes":summary["root_native_counterfactual_sha256"],
         "decision":("weighted_higher" if summary["weighted_minus_uniform"]>0 else
                     "uniform_higher" if summary["weighted_minus_uniform"]<0 else "tied"),
         "whole_opening_games_added":0,"runner":summary["runner_sha256"],"summary_sha256":common.sha((d/"summary.json").read_bytes())}
    common.save(d/"verified.json",out)
    print(json.dumps(out,ensure_ascii=False))


if __name__=="__main__":
    main()

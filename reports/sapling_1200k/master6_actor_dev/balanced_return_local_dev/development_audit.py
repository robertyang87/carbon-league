"""Independently replay the local policy on the four archived, explicitly exposed proxy trajectories."""
import argparse
import hashlib
import sys
import time

from audit import HERE, PREVIOUS, WINDOWS, check, common, read, sha, worlds


def key(obj):
    return hashlib.sha256(common.encode(obj)).hexdigest()


def opening(pa, ca, seat):
    limit=min(len(ca),len(pa))
    first=next((t for t in range(limit) if pa[t]!=ca[t]),None)
    if first is None:
        check(len(ca)==len(pa),"same trajectory has two different terminal lengths")
    else:
        check(first in WINDOWS and pa[first][seat]!=ca[first][seat] and pa[first][1-seat]==ca[first][1-seat],
              "first divergence is outside the gate or the proxy anticipated a simultaneous move")
    return first


def run():
    common.assert_cpu()
    out=HERE/"development"
    r=read(HERE/"release.json");protocol=read(HERE/"protocol.json");mech=read(HERE/"mechanism.json")
    registered=read(PREVIOUS/"development/comparison.json");latest=read(out/"comparison.json")
    check(sha(PREVIOUS/"development/comparison.json")==protocol["references"]["historical_comparison"] and len(latest["pairs"])==4,"wrong reference block")
    check(sha(HERE/"candidate.py")==r["candidate_sha256"]==mech["candidate_sha256"] and sha(common.PARENT)==r["parent_sha256"],"policy identity")
    parent=common.policy.load(common.PARENT);local=common.policy.load(HERE/"candidate.py")
    opponent=common.net("frozen_it50")
    old={(p["seed"],p["seat"]):p for p in registered["pairs"]}
    current={(p["seed"],p["seat"]):p for p in latest["pairs"]}
    schedule={(s,p) for s in protocol["maps"] for p in (0,1)}
    check(set(old)==set(current)==schedule,"incomplete paired map or seat set")
    summaries=[]
    for s,p in sorted(schedule):
        cf=out/f"candidate_{s}_{p}.json.gz";pf=PREVIOUS/"development"/f"parent_{s}_{p}.json.gz"
        c,past=read(cf),read(pf)
        check(sha(pf)==old[s,p]["arms"]["parent"]["tape_sha256"]==current[s,p]["arms"]["parent"]["tape_sha256"]
              and sha(cf)==current[s,p]["arms"]["local"]["tape_sha256"],"game not in both sealed comparison blocks")
        check(c["ticket"]["binding"]==latest["binding"] and past["ticket"]["binding"]==registered["binding"],"replay lineage")
        check(c["ticket"]["binding"]["opponent_sha256"]==past["ticket"]["binding"]["opponent_sha256"]==sha(common.IT50)
              and c["ticket"]["terrain_sha256"]==past["ticket"]["terrain_sha256"]==worlds.locked_terrain(s,"development"),"opponent or terrain differs")
        actions=c["actions"];pa=past["actions"]
        check(key(actions)==c["actions_sha256"] and key(pa)==past["actions_sha256"],"changed command tape")
        first=opening(pa,actions,p)
        game=worlds.World(s,"development");changed=[];student_samples=0;student_changed=0
        without=common.policy.load(HERE/"candidate.py");without._HYBRID["student_weight"]=0.
        for t,(pair,shadow) in enumerate(zip(actions,c["shadow"],strict=True)):
            own=game.obs(p);foe=game.obs(1-p)
            a=local.agent(own,worlds.CFG);b=parent.agent(own,worlds.CFG);enemy=opponent.agent(foe,worlds.CFG)
            check((shadow["physical_seat"],shadow["step"])==(p,t) and shadow["public_observation_sha256"]==key(own),"shadow observed a different physical state")
            check(a==pair[p]==shadow["candidate"] and b==shadow["parent"] and enemy==pair[1-p],"issued command differs on a complete observed history")
            if a!=b:
                check(t in WINDOWS,"local policy changed command outside action windows");changed.append(t)
            if t%10==0:
                no=without.agent(own,worlds.CFG)
                check(no==shadow["candidate_without_sapling"],"student ablation disagrees")
                student_samples+=1;student_changed+=int(no!=a)
            else:without._FEAT.observe(own)
            game.advance(pair)
        check(game.terminal is not None and len(actions)==c["official"]["steps"],"truncated official game")
        off=worlds.replay_official(s,actions,"development")
        check(off==c["official"],"new official score differs")
        check((first is None)==(not changed) and (first is None or first==changed[0]),"paired and shadow first divergence disagree")
        if first is None:
            check(c["official"]==past["official"] and key(actions)==key(pa),"identical commands did not reproduce the same official end")
        summaries.append({"seed":s,"seat":p,"steps":len(actions),"independently_verified_commands":2*len(actions),
                          "same_root_parent_local_changed_steps":changed,"first_joint_difference_to_archived_parent":first,
                          "identical_joint_prefix_steps":len(actions) if first is None else first,
                          "equal_full_joint_tape":c["actions_sha256"]==past["actions_sha256"],
                          "student_ablation_samples":student_samples,"independent_student_effects":student_changed,
                          "candidate_tape_sha256":sha(cf),"parent_tape_sha256":sha(pf),
                          "official_parent_margin":past["official"]["margin_by_seat"][p],"official_local_margin":off["margin_by_seat"][p]})
    result={"scope":"all four previously exposed proxy cases with verified full openings and same-observation native policy calls; no new match","formal_games":0,
            "original_master3s2_games":0,"comparison_sha256":sha(out/"comparison.json"),"mechanism_sha256":sha(HERE/"mechanism.json"),
            "audit_sha256":sha(HERE/"development_audit.py"),"cases":summaries,"steps":sum(x["steps"] for x in summaries),
            "actual_joint_changes":sum(len(x["same_root_parent_local_changed_steps"]) for x in summaries),
            "identical_complete_paired_tapes":sum(x["equal_full_joint_tape"] for x in summaries),
            "student_ablation_changes":sum(x["independent_student_effects"] for x in summaries)}
    common.write(out/"mechanism_audit.json",result);print({k:v for k,v in result.items() if k!="cases"},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true");args=parser.parse_args()
    if args.wait:
        while not (HERE/"development/comparison.json").is_file():time.sleep(2)
    run()

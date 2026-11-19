"""Check which controlled alternatives were already selected by the frozen orchard agent."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import orchard_branches as b
from orchard_high_branches import OUT


def audit(file,source):
    g=json.loads(gzip.decompress(file.read_bytes()))
    root=g["root"]; seat=g["card"]["seat"]; step=root["step"]
    game=json.loads(gzip.decompress(source.read_bytes()))
    if hashlib.sha256(b.canonical(game)).hexdigest()!=g["card"]["canonical_sha256"]:
        raise ValueError("wrong historical root")
    frames=game["steps"]; cfg=game["configuration"]
    initial=frames[0][0]["observation"]
    sim=b.arena.exam.FastCarbon(cfg).reset(full_carbon=initial["full_carbon"],cash=initial["players"][0][0])
    m=b.arena.prior.shared.module(b.PACK,"full_orchard_replay")
    m.agent(b.pub(sim.obs(seat)),cfg)
    for t in range(1,step+1):
        sim.step(*(player["action"] or {} for player in frames[t]))
        if t<step:
            m._FEAT.observe(b.pub(sim.obs(seat)))
    if b.arena.prior.hashed(sim.get_state())!=root["root_state_sha256"]:
        raise ValueError("wrong action root")
    prev=m._orchard_decide
    detail={}
    def capture(F,x,prefix,slot,normal,logits,legal,changes):
        if slot==root["slot"]:
            detail.update(prefix_cash=float(prefix.cash), reserve_after_buy=float(prefix.cash)-root["public_price"],
                          policy_logit_gap_to_stay=float(logits[normal]-logits[0]), allowed_gap=min(2.,root["predicted_net"]/100.),
                          committed_earlier_change=bool(changes), standard_action=int(normal),stay_legal=bool(legal[0]))
        return prev(F,x,prefix,slot,normal,logits,legal,changes)
    m._orchard_decide=capture
    action=m.agent(b.pub(sim.obs(seat)),cfg)
    buy,move=g["buy"],g["move"]
    if buy["actions"][0][1-seat]!=move["actions"][0][1-seat]:
        raise ValueError("opposing root command changed")
    changed={k for k in buy["actions"][0][seat].keys()|move["actions"][0][seat].keys()
             if buy["actions"][0][seat].get(k)!=move["actions"][0][seat].get(k)}
    if changed!={root["unit"]} or not buy["root_tree"]:
        raise ValueError("root did not exclusively change and settle the intended purchase")
    return {"id":g["card"]["id"],"seat":seat,"group":g["card"]["group"],"step":step,
            "root_economics":detail,"frozen_orchard_matches_move":action==move["actions"][0][seat],
            "frozen_orchard_matches_buy":action==buy["actions"][0][seat],"predicted_net":root["predicted_net"],
            "actual_tree_net":buy["root_tree"]["net"], "buy_minus_move":g["buy_minus_move"],
            "buy_margin":buy["margin"],"move_margin":move["margin"],
            "own_delivery_change":buy["by_seat"][seat].get("own_delivery",0.)-move["by_seat"][seat].get("own_delivery",0.),
            "enemy_delivery_change":buy["by_seat"][1-seat].get("own_delivery",0.)-move["by_seat"][1-seat].get("own_delivery",0.)}


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--replays",type=Path,required=True)
    args=ap.parse_args()
    if b.arena.prior.digest(b.PACK)!=b.SHA:raise ValueError("frozen pack differs")
    paths={p.name.removesuffix(".json.gz"):p for p in args.replays.rglob("*.json.gz")}
    coverage=json.loads((OUT/"coverage.json").read_bytes())
    rows=[audit(OUT/(g["chosen"]+".json.gz"),paths[g["chosen"]]) for lane in coverage for g in lane.values() if g["chosen"]]
    b.write(OUT/"summary.json",{"examined":sum(len(g["examined"]) for lane in coverage for g in lane.values()),
                               "frozen_maps":24,"roots":rows})

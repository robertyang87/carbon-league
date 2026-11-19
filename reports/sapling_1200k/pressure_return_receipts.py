"""Follow actual stolen units through settlement; replay the unmodified root with the same enemy command."""
import gzip
import json
from pathlib import Path
import sys
import tree_audited_game as arena
from turning_point_ledger import CashEvents

HERE=Path(__file__).resolve().parent;OUT=HERE/"all_top5_pressure_dev"
sys.path.insert(0,str(HERE/"all_top5"))
from defense import audit


def physical_frame(sim,orders):
    board=sim.obs(0)
    board["full_carbon"]=list(sim.carbon)
    return [{"observation":board,"action":orders[0]},{"observation":sim.obs(1),"action":orders[1]}]


def round_out(sim,item,orders,seat):
    s=arena.exam.FastCarbon(sim.config);s.set_state(sim.get_state());s.cash=CashEvents(s.cash)
    s.step(*orders)
    v=item["victim"];h=item["hunter"]
    collision=next((r for r in s.last_collisions if r[0]==v),None)
    own_del=sum(e[2] for e in s.cash.events if e[0]=="own_delivery" and e[1]==seat)
    opp_del=sum(e[2] for e in s.cash.events if e[0]=="own_delivery" and e[1]==1-seat)
    return {"victim_dead":v not in s.workers,"hunter_alive":h in s.workers,
            "cargo_left_victim_in_collision":collision[2] if collision else 0.,
            "recipient_seat_in_collision":collision[3] if collision else None,
            "own_normal_delivery":own_del,"enemy_normal_delivery":opp_del}


def one(label,game,changes):
    p=OUT/label/game["raw_file"]
    if arena.prior.digest(p)!=game["raw_sha256"]:raise ValueError("wrong action tape")
    raw=json.loads(gzip.decompress(p.read_bytes()));seat=raw["candidate_seat"]
    sim=arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    frames=[physical_frame(sim,[{},{}])];comp=[]
    for t,commands in enumerate(raw["actions"]):
        if t in changes:
            item=changes[t];alt=[dict(cmd) for cmd in commands]
            if item["parent"]:alt[seat][item["hunter"]]=item["parent"]
            else:alt[seat].pop(item["hunter"],None)
            comp.append({"step":t,"victim":item["victim"],"hunter":item["hunter"],
                         "actual":round_out(sim,item,commands,seat),
                         "one_step_parent":round_out(sim,item,alt,seat)})
        sim.step(*commands)
        frames.append(physical_frame(sim,commands))
    if arena.prior.hashed(sim.get_state())!=game["final_state_sha256"]:raise ValueError("terminal state mismatch")
    pkt=audit({"configuration":raw["game_configuration"],"steps":frames})
    keys={(r["step"]+1,r["hunter"],r["victim"]) for r in comp}
    lots=[l for l in pkt["lots"] if l["method"]=="collector" and l["beneficiary"]==seat]
    affected=[l for l in lots if (l["t"],l["hunter"],l["victim"]) in keys]
    return {"seed":raw["seed"],"seat":seat,"total_stolen_collector_cash_lots":len(lots),
            "total_collector_cargo_taken":sum(l["mass"] for l in lots),
            "total_collector_takings_later_banked":sum(l["fate"].get("banked",0.) for l in lots),
            "target_lots_after_changed_roots":affected,"same_step_counterfactual":comp}


if __name__=="__main__":
    attr=json.loads((OUT/"candidate/attribution.json").read_bytes());d={}
    by_seat={g["seat"]:{r["step"]:r for r in g["changes"]} for g in attr["games"]}
    for flag in ("candidate","parent"):
        games=json.loads((OUT/flag/"results.json").read_bytes())["games"]
        d[flag]=[one(flag,g,by_seat[g["candidate_seat"]] if flag=="candidate" else {}) for g in games]
    (OUT/"capture_receipts.json").write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")

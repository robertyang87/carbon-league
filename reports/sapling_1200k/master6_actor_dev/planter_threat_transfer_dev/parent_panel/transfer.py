"""Replay the unchanged pay selector on the excluded-terrain complement. Compare one official step per arm."""
from collections import Counter
from copy import deepcopy
import json
import sys

from panel import ACTOR,BLOCKED,HERE,PAY,ROOT,check,digest,read,save,sha

sys.path.insert(0,str(PAY))
import opportunities as old_pay
sys.path.insert(0,str(ACTOR/"lot_settlement_dev"))
import lots

F=old_pay.FastCarbon
MOVES=("NONE","UP","RIGHT","DOWN","LEFT")
SHIFT=((0,0),(-1,0),(0,1),(1,0),(0,-1))


def step(p,a):
    y,x=divmod(p,15);dy,dx=SHIFT[MOVES.index(a)]
    return ((y+dy)%15)*15+(x+dx)%15


def distance(a,b):
    y,x=divmod(a,15);v,w=divmod(b,15)
    return min(abs(y-v),15-abs(y-v))+min(abs(x-w),15-abs(x-w))


def tree_state(players,p):
    trees=[{"side":s,"uid":u,"age":age} for s,pl in enumerate(players) for u,(cell,age) in pl[3].items() if cell==p]
    check(len(trees)<=1,"two trees on same cell")
    return trees[0] if trees else None


def exposure(view, cfg, seat, cell):
    enemy=view["players"][1-seat]
    possibly=[]
    for u,(p,c,role) in sorted(enemy[2].items()):
        moves=[a for a in MOVES if step(p,a)==cell]
        if moves:
            possibly.append({"uid":u,"cargo":c,"role":role,"from":p,"possible_moves":moves})
    role_counts=Counter(w[2] for pl in view["players"] for w in pl[2].values())
    spawns=[]
    if len(enemy[2])<cfg["workerLimit"]:
        for b,p in enemy[1].items():
            if cell==p:
                for role,key in (("COLLECTOR","Collector"),("PLANTER","Planter")):
                    required=cfg["rec"+key+"Cost"]+cfg["rec"+key+"IncreaseCost"]*role_counts[role]
                    if enemy[0]>=required:
                        spawns.append([b,role,required])
    return {"existing":possibly,"possible_enemy_recruit_on_cell":spawns,
            "nearest":min((distance(w[0],cell) for w in enemy[2].values()),default=None)}


def official_step(sim, orders):
    before=sim.full_observation()
    lots.idgen._id_dict.clear()
    for p in (0,1):
        lots.idgen._id_dict[f"player-{p}-worker-"]=sim.worker_counters[p]
        lots.idgen._id_dict[f"player-{p}-tree-"]=sim.tree_counters[p]
    after=lots.Board(before,sim.config,orders).next().observation
    sim.step(*orders)
    reproduced=sim.full_observation()
    check(all(after[k]==reproduced[k] for k in ("players","trees","carbon")),("official/fast disagreement",before["step"]))
    return after


def arm(sim,joint,seat,uid,plant_cell):
    branch=F(sim.config)
    branch.set_state(sim.get_state())
    branch.cash=old_pay.CashEvents(list(branch.cash))
    before=branch.full_observation()
    draws=branch.cash
    dest=step(before["players"][seat][2][uid][0],joint[seat].get(uid,"NONE"))
    pre_exposure=exposure(before,sim.config,seat,dest)
    enemy_arrivals=[]
    ally_arrivals=[]
    for who,pl in enumerate(before["players"]):
        for u,(origin,cargo,role) in pl[2].items():
            if u==uid:
                continue
            order=joint[who].get(u,"NONE")
            if step(origin,order)==dest:
                (ally_arrivals if who==seat else enemy_arrivals).append({"uid":u,"role":role,"from":origin,"order":order,"cargo":cargo})
    after=official_step(branch,joint)
    events=list(draws.events)
    delta=[Counter(),Counter()]
    for kind,who,v,*_ in events: delta[who][kind]+=v
    for side in (0,1):
        check(abs(sum(delta[side].values())-(after["players"][side][0]-before["players"][side][0]))<1e-5,"cash imbalance")
    recv=[]
    for kind,who,v,*attr in events:
        if kind in ("plant","seize") and who==seat and attr[0]==uid:
            recv.append([kind,-v,attr[1]])
    check(len(recv)<=1,"multiple debits to focal")
    hires=[{"side":who,"role":kind,"cost":-v,"base":base,"pos":p} for kind,who,v,base,p in
           (e for e in events if e[0].startswith("hire_"))]
    player_after=after["players"][seat][2].get(uid)
    return {"actual_order":joint[seat].get(uid,"NONE"),"actual_target":dest,"pre_reach":pre_exposure,
            "enemy_real_existing_arrivals":enemy_arrivals,"friendly_real_existing_arrivals":ally_arrivals,
            "actual_hires":hires,"enemy_spawned_at_target":[h for h in hires if h["side"]==1-seat and h["pos"]==dest],
            "survived":player_after is not None,"worker_after":player_after,"worker_debit":recv[0][1] if recv else 0.,
            "booked":recv,"tree_at_proposed_cell_before":tree_state(before["players"],plant_cell),
            "tree_at_proposed_cell_after":tree_state(after["players"],plant_cell),
            "tree_at_actual_target_after":tree_state(after["players"],dest),"cash_before":[p[0] for p in before["players"]],
            "cash_after":[p[0] for p in after["players"]],"cash_categories_by_side":list(map(dict,delta)),
            "official_after_sha256":digest({k:after[k] for k in ("players","trees","carbon")})}


def validate_source(row,plan):
    check(row["seed"] not in BLOCKED,"new pilot blocked before opening any tape")
    p=ROOT/row["raw"]; cert=ROOT/row["official"]
    check(sha(p)==row["raw_sha256"] and sha(cert)==row["official_sha256"],"signed source changed")
    tape=read(p); official=read(cert)
    check(tape["seed"]==row["seed"] and tape["candidate_seat"]==row["seat"] and
          tape["initial_terrain_u8_sha256"]==row["map"] and row["map"] not in plan["excluded_terrains"]
          and tape["pack_sha256"]==row["pack_sha256"]==plan["parent_sha256"]
          and tape["examiner_md5"]==row["examiner_md5"] and len(tape["actions"])==tape["steps"]==row["steps"]
          and not tape["errors"] and not tape["audit_messages"],"wrong trajectory identity")
    certified=[g for a in official["runs"].values() if a["pack_sha256"]==row["pack_sha256"] for g in a["games"]
               if (g["seed"],g["seat"])==(row["seed"],row["seat"])]
    check(len(certified)==1 and certified[0]["transcript_sha256"]==row["raw_sha256"] and
          certified[0]["official_reward_by_seat"]==tape["official_score_by_seat"] and
          certified[0]["statuses"]==["DONE","DONE"],"official source not completed")
    check(tape["configuration"]["_S_ENGINE"] is True and tape["configuration"]["_TEMP"]==1.0 and
          tape["configuration"]["_SEED"] is None,"examiner settings")
    return tape


def game(row,plan,pack):
    tape=validate_source(row,plan);cfg=tape["game_configuration"];s=row["seat"]
    sim=F(cfg).reset(seed=row["seed"])
    corners=[(45,125),(150,235)]
    reached=[sum(t<len(tape["actions"]) for t in range(lo,hi+1)) for lo,hi in corners]
    cases=[]; inspected=[0,0]; prefix=0
    for t,joint in enumerate(tape["actions"]):
        if t>235 or (t and len(pack._PAY_USED)==2):
            break
        obs=sim.obs(s)
        window=next((w for w,(lo,hi) in enumerate(corners) if lo<=t<=hi),None)
        if t==0 or (window is not None and window not in pack._PAY_USED):
            cursor=0 if t==0 else len(pack._PAY_TRACE)
            out=pack.agent(obs,cfg)
            if window is not None:
                inspected[window]+=1
            if len(pack._PAY_TRACE)>cursor:
                tr=deepcopy(pack._PAY_TRACE[-1]);base=tr["base"]
                check(tr["t"]==t and tr["window"]==window and base==joint[s],"selection differs from signed true parent")
                uid=tr["uid"];cell=tr["cell"]
                check(uid in obs["players"][s][2] and obs["players"][s][2][uid]==[cell,0,"PLANTER"],"unexpected target worker")
                pay_order=dict(base);pay_order.pop(uid,None)
                check(out==pay_order and tr["changed"]==(base!=out),"candidate is not the registered payment")
                actual=arm(sim,joint,s,uid,cell)
                changed=[dict(x) for x in joint];changed[s]=out
                pay=arm(sim,changed,s,uid,cell)
                check(not pay["worker_debit"] or (pay["survived"] and pay["booked"][0][0]==tr["kind"] and
                      pay["tree_at_proposed_cell_after"]["side"]==s),"debited without acquired tree")
                cases.append({"window":window,"t":t,"uid":uid,"cell":cell,"kind":tr["kind"],"changed":tr["changed"],
                              "public_input_sha256":digest(obs),"quoted_cost":tr["quoted_cost"],"pre_action_own_cash":tr["own_cash"],
                              "other_planter_reserve":tr["other_planter_reserve"],"hire_reserve":tr["hire_reserve"],
                              "public_reach":exposure(obs,cfg,s,cell),"enemy_submitted":dict(joint[1-s]),"parent":actual,"pay":pay})
            else:
                check(out==joint[s],"public selector changed a non-selected root")
        else:
            pack._FEAT.observe(obs)
        official_step(sim,joint)
        prefix+=1
    return {"seed":row["seed"],"map":row["map"],"seat":s,"source_sha256":row["raw_sha256"],
            "trace_steps":row["steps"],"official_parent_prefix_steps":prefix,"available_windows":sum(n>0 for n in reached),
            "available_turns_per_window":reached,"chronological_inspected_until_selection":inspected,"selected":cases}


def main():
    plan=read(HERE/"panel.json")
    check(sha(PAY/"candidate.py")==plan["pay_candidate_sha256"] and sha(PAY/"protocol.json")==plan["pay_protocol_sha256"],
          "selector changed after freeze")
    pack=old_pay.pack()
    for entry in plan["games"]:
        out=HERE/"games"/f'{entry["seed"]}_{entry["seat"]}.json.gz';out.parent.mkdir(exist_ok=True)
        fingerprint={"panel_sha256":sha(HERE/"panel.json"),"runner_sha256":sha(HERE/"transfer.py")}
        if out.exists():
            old=read(out)
            check(old["identity"]==fingerprint and old["game"]["source_sha256"]==entry["raw_sha256"],"stale result")
            continue
        g=game(entry,plan,pack)
        save(out,{"identity":fingerprint,"game":g})
        print(entry["seed"],entry["seat"],[(s["kind"],s["changed"],bool(s["public_reach"]["existing"]),s["pay"]["survived"]) for s in g["selected"]],flush=True)


if __name__=="__main__":
    main()

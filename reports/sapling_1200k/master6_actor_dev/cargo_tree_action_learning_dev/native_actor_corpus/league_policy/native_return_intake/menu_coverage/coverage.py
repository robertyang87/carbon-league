"""Replay both logged joint action choices against the opponent's actual simultaneous order."""
import copy
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
ROOT = next(p for p in HERE.parents if (p/".git").exists())
SAP = ROOT/"reports/sapling_1200k"
sys.path.insert(0, str(INTAKE/"endgame_curriculum"))
sys.path.insert(0, str(SAP))
import roots
from turning_point_ledger import FastCarbon, CashEvents, tree_changes

MANIFEST = "63daa51e127aafea36401ebdce6400de60faa772d3929a1803ce768e121be5b4"
KEYS = (
    "plants", "seizures", "tree_expiries", "collector_hires", "planter_hires",
    "recruit_cash", "plant_cash", "seizure_cash", "warehouse_deposit", "gift_received",
    "tree_ground_income", "tree_enemy_cargo_tax", "tree_collision_cash", "base_collision_cash",
    "own_root_loaded_deaths", "own_root_load_on_dead", "opponent_cargo_looted",
    "own_root_load_lost_to_enemy_collector", "own_root_load_still_alive", "closing_worker_cargo",
    "cash_delta")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def rounded(number):
    return round(float(number), 6)


def check_board(sim, env):
    o=env.states[0].observation
    obs=sim.obs(0)
    need(sim.step_count==o.step and np.allclose(sim.carbon,o.full_carbon,rtol=0,atol=1e-7), "step/ground differs")
    for short,long in zip(obs["players"],o.players):
        need(abs(short[0]-long[0])<1e-7 and short[1]==long[1] and short[3]==long[3] and
             short[2].keys()==long[2].keys(),"cash, base, tree or ID differs")
        for uid,(pos,load,role) in short[2].items():
            a,b,c=long[2][uid]
            need(pos==a and role==c and abs(load-b)<1e-7,"worker differs")
    need(obs["trees"].keys()==o.trees.keys(),"tree income roster differs")
    for tid,(owner,cash) in obs["trees"].items():
        own,c=o.trees[tid]
        need(owner==own and abs(cash-c)<1.e-7,"tree payment differs")
    from zerosum_env.envs.carbon import idgen
    for s in (0,1):
        need([sim._tid_counter[s],sim._wid_counter[s]]==
             [idgen._id_dict.get(f"player-{s}-tree-",0),idgen._id_dict.get(f"player-{s}-worker-",0)],
             "official global allocation counters differ")


def account(sim, orders, root):
    pre_cash = list(sim.cash)
    workers = root["workers"]
    sim.cash=CashEvents(sim.cash)
    sim.step(*orders)
    balance = [{k:0. for k in KEYS} for _ in range(2)]
    events = sim.cash.events
    for kind,seat,amount,*d in events:
        z=balance[seat]
        if kind.startswith("hire_"):
            z["collector_hires" if kind=="hire_collector" else "planter_hires"]+=1
            z["recruit_cash"]+=amount
        elif kind=="tree_absorption":
            # This exact cash write contains both ground and enemy carrier tax.
            z["tree_ground_income"]+=amount-d[1];z["tree_enemy_cargo_tax"]+=d[1]
        else:
            key={"plant":"plant_cash","seize":"seizure_cash","own_delivery":"warehouse_deposit",
                 "opponent_worker_delivery_to_this_base":"gift_received","tree_collision_cash":"tree_collision_cash",
                 "base_collision_cash":"base_collision_cash"}.get(kind)
            if key:z[key]+=amount
    changes=tree_changes({tid:(v["player"],v["pos"]) for tid,v in root["trees"].items()},sim)
    for kind,tid,pos,before,after in changes:
        if kind=="new":balance[after]["plants"]+=1
        elif kind=="seizure":balance[after]["seizures"]+=1
        elif kind=="expiry":balance[before]["tree_expiries"]+=1
    for uid,owner,cargo,payee,collector in sim.last_collisions:
        if collector is not None and owner!=collector:
            balance[collector]["opponent_cargo_looted"]+=cargo
            if uid in workers:
                balance[owner]["own_root_load_lost_to_enemy_collector"]+=min(workers[uid]["carbon"],cargo)
    for uid,w in workers.items():
        if w["type"]=="COLLECTOR" and w["carbon"]>0:
            z=balance[w["player"]]
            if uid in sim.workers:
                z["own_root_load_still_alive"]+=w["carbon"]
            else:
                z["own_root_loaded_deaths"]+=1;z["own_root_load_on_dead"]+=w["carbon"]
    for side,z in enumerate(balance):
        z["closing_worker_cargo"]=sum(w.carbon for w in sim.workers.values() if w.player==side and w.is_collector)
        z["cash_delta"]=sim.cash[side]-pre_cash[side]
        need(abs(z["cash_delta"]-sum(e[2] for e in events if e[1]==side))<1e-6,"unattributed cash")
    # Retain payment owners, worker/tree IDs and actual amounts per individual event.
    # Keep tree owner/position edits as independent mechanical evidence of taken/replanted trees.
    return dict(by_seat=[{k:rounded(v) for k,v in z.items()} for z in balance],
                tree_changes=changes,cash_events=[e for e in events if e[0]!="rounding"],
                collisions=sim.last_collisions)


def episode(ep):
    ref=dict(path=ep["archive_path"],sha256=ep["archive_sha256"],seed=ep["seed"],seat=ep["seat"])
    seat=ep["seat"];source={e["step"]:e for e in ep["events"]};out=[]
    with roots.open_episode(ref,ep["arm"]) as (env,live,tape,registration,reg_bytes):
        from zerosum_env.envs.carbon import idgen
        need(sha(reg_bytes)==ep["registration_sha256"],"registration changed")
        sim=FastCarbon(env.configuration).reset(seed=ep["seed"])
        check_board(sim,env)
        bank=[5.,5.]; tape_hash=hashlib.sha256()
        for t,line in enumerate(tape):
            tape_hash.update(line)
            entry=json.loads(line)
            need(t==entry["step"] and all(c["status"]=="ok" for c in entry["calls"]),"bad source step")
            hashes=[sha(live.encoded(live.observe(env,p,bank[p]))) for p in (0,1)]
            need(hashes==entry["public_observation_sha256"],"different replay prefix")
            orders=[c["response"]["action"] for c in entry["calls"]]
            ev=source.get(t)
            if ev and ev["supported"]:
                need(ev["public_observation_sha256"]==hashes[seat] and
                     ev["options"][ev["choice"]]==orders[seat] and ev["options"][0]!=ev["options"][1],"menu identity")
                root=sim.get_state();ids_before=dict(idgen._id_dict)
                both=[]
                for our_order in ev["options"]:
                    joint=orders.copy();joint[seat]=our_order
                    branch=copy.copy(env)
                    branch.states=copy.deepcopy(env.states)
                    branch.steps=env.steps[:-1]+[branch.states];branch.logs=[];branch.error=env.error.copy()
                    branch.step(joint,[])
                    reference=FastCarbon(env.configuration).set_state(root)
                    ans=account(reference,joint,root)
                    check_board(reference,branch)
                    both.append(ans)
                    idgen._id_dict.clear();idgen._id_dict.update(ids_before)
                components=[]
                for uid in sorted(ev["options"][0].keys()|ev["options"][1].keys()):
                    a,b=[cmd.get(uid,"NONE") for cmd in ev["options"]]
                    if a==b:continue
                    w=root["workers"].get(uid)
                    components.append(dict(uid=uid,role=w["type"] if w else "BASE",pos=w["pos"] if w else
                                           dict(root["bases"][seat])[uid],before_load=w["carbon"] if w else None,orders=[a,b]))
                out.append(dict(seed=ep["seed"],seat=seat,arm=ep["arm"],step=t,terrain=ep["terrain"],choice=ev["choice"],
                                probabilities=ev["probabilities"],focal_uid=ev["uid"],changed_components=components,
                                options=ev["options"],recorded_opponent_order=orders[1-seat],public_observation_hashes=hashes,
                                metrics_by_option=both))
            sim.step(*orders);env.step(orders,[])
            if ev and ev["supported"]:check_board(sim,env)
            bank=entry["overage_remaining_seconds"]
        need(tape_hash.hexdigest()==ep["tape_sha256"] and env.done and env.states[0].observation.step==ep["official_transitions"]
             and [float(s.reward) for s in env.states]==ep["official_scores_by_seat"],"source ended differently")
        check_board(sim,env)
    return out


def summarize(rows):
    def agg(selected):
        result={}
        for key in KEYS:
            pairs=[[r["metrics_by_option"][a]["by_seat"][r["seat"]][key] for a in (0,1)] for r in selected]
            diff=[rounded(v[1]-v[0]) for v in pairs]
            result[key]=dict(changed=sum(abs(x)>1e-5 for x in diff),any_positive=sum(any(abs(a)>1e-5 for a in p) for p in pairs),
                             alternative_greater=sum(x>1e-5 for x in diff),alternative_lower=sum(x< -1e-5 for x in diff),
                             min_delta=min(diff,default=None),max_delta=max(diff,default=None))
        both_paid=Counter()
        for r in selected:
            seat=r["seat"];opts=r["metrics_by_option"]
            for key in KEYS:
                if abs(opts[0]["by_seat"][1-seat][key]-opts[1]["by_seat"][1-seat][key])>1e-5:both_paid[key]+=1
        return dict(roots=len(selected),own=result,enemy_changed=dict(both_paid))
    roles={r:sum(sum(c["role"]==r for c in z["changed_components"]) for z in rows) for r in ("COLLECTOR","PLANTER","BASE")}
    return dict(all=agg(rows),physical_command_edits=roles,
                focal_role={role:agg([r for r in rows if any(c["role"]==role for c in r["changed_components"])]) for role in roles},
                maps=[dict(seed=s,**agg([r for r in rows if r["seed"]==s])) for s in sorted({r["seed"] for r in rows})])


def main():
    path=INTAKE/"hosted/training_manifest.json";raw=path.read_bytes();need(sha(raw)==MANIFEST,"training manifest changed")
    d=json.loads(raw);eps=d["episodes"];need(len(eps)==24 and len({(e["seed"],e["seat"],e["arm"]) for e in eps})==24,"missing source")
    allrows=[]
    for ep in eps:
        rr=episode(ep);allrows+=rr;print(ep["seed"],ep["seat"],ep["arm"],len(rr),flush=True)
    need(len(allrows)==275,"support census differs")
    result=dict(manifest_sha256=MANIFEST,settlement_sha256=d["settlement_sha256"],actor=d["policy_sha256"],referee=d["examiner_sha256"],
                registered_draws=288,unsupported=13,verified_native_game_trajectories=len(eps),verified_official_branch_transitions=2*len(allrows),
                count_by_role_is_overlapping=True,summary=summarize(allrows),rows=allrows)
    raw=json.dumps(result,separators=(",",":"),sort_keys=True,allow_nan=False).encode()
    out=HERE/"one_step.json.gz";tmp=out.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(raw,mtime=0));tmp.replace(out)


if __name__=="__main__":
    main()

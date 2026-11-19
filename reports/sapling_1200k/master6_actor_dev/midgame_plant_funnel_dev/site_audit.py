"""Attribute an abstention to individual own-public feature tests on the two frozen roots."""
import importlib.util
from pathlib import Path
import runpy
trial = runpy.run_path(str(Path(__file__).resolve().with_name("trial.py")))
HERE, sr, PACK, W = (trial[k] for k in ("HERE","sr","PACK","W"))
from route import _D, _DIRECTION, _TO, mode
from decision import predicted, quote, update_memory, values

def diagnostics(obs,mem,orders,uid,cell,cfg):
    s=obs["player"];o=obs["players"][s];e=obs["players"][1-s]
    trees={int(p):c for c,side in enumerate(obs['players']) for p,a in side[3].values()}
    bases={p for side in obs["players"] for p in side[1].values()}
    now=[k for side in obs["players"] for k,w in side[2].items() if k!=uid and w[0]==cell]
    planned=[k for k,w in o[2].items() if k!=uid and _TO[w[0],_DIRECTION.index(orders.get(k))]==cell]
    danger=sorted([dict(id=k,role=w[2],distance=int(_D[cell,w[0]]),position=w[0]) for k,w in e[2].items()],key=lambda v:(v["distance"],v["id"]))
    reserve=0.; fee=quote(obs,cfg)
    for k,(p,_,role) in o[2].items():
        if k!=uid and role=="PLANTER" and not orders.get(k) and p not in bases:
            reserve+=fee if p not in trees else (cfg["seizeCost"] if trees[p]!=s else 0.)
    for base in o[1]:
        order=orders.get(base)
        if order:
            kind=order[3:].title()
            n=sum(w[2]==kind.upper() for side in obs['players'] for w in side[2].values())
            reserve+=cfg["rec"+kind+"Cost"]+n*cfg["rec"+kind+"IncreaseCost"]
    reasons=[]
    if cell in bases:reasons.append("base")
    if cell in trees:reasons.append("living_tree")
    if now:reasons.append("another_worker_currently_occupies")
    if planned:reasons.append("another_friendly_worker_will_enter")
    if danger and danger[0]["distance"]<=1:reasons.append("enemy_can_enter_next_move")
    if o[0]<fee+reserve:reasons.append("insufficient_unreserved_cash")
    assert (not reasons)==mode(obs,orders,uid,cell,cfg)
    f=values(obs,mem,uid,cell,cfg)
    observed=[int(p) for p in _TO[cell][1:] if p in mem and mem[p][1]==obs["step"]]
    return dict(cell=cell,accessible_neighbor_observations=len(observed),current_view_observed_sum=sum(obs["carbon"][p] for p in observed),
        estimated_ground_carbon_sum=f[2]*100 if f else None, known_fraction=f[3] if f else None,
        net_ground=predicted(f,W)-fee if f else None,
        reasons=reasons,cash=o[0],reserved=reserve,fee=fee,nearest_enemy=next(iter(danger),None),
        current_occupants=now,intended_friendly_occupants=planned)

def audit():
    rows=[]
    for q in sr.load(HERE/'selected_sites.json'):
        raw=sr.load(sr.ROOT/q["record"]["source"]);cfg=raw["game_configuration"];t=q["root_t"];s=q["seat"]
        spec=importlib.util.spec_from_file_location('site_audit_parent',PACK/'candidate.py')
        parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
        sim=sr.FastCarbon(cfg).reset(seed=q["seed"]);mem={};root=q["root"];uid=root["worker"]
        for a,pair in enumerate(raw["actions"][:t+1]):
            obs=sim.obs(s);update_memory(obs,mem)
            orders=parent.agent(obs,cfg)
            if orders!=pair[s]:raise ValueError("deterministic source mismatch")
            if a==t:
                options=[]
                for direction,dest in zip(_DIRECTION[1:],_TO[root["cell"]][1:]):
                    d=diagnostics(obs,mem,orders,uid,int(dest),cfg);d["direction"]=direction
                    d["distinct_qualified_alternative"]=(int(dest)!=root["prior_destination"] and not d["reasons"]
                        and d["known_fraction"] is not None and d["known_fraction"]>=.5 and d["estimated_ground_carbon_sum"]>=120
                        and d["net_ground"]-root["alternative_prior_net"]>10)
                    options.append(d)
                altered=list(pair);altered[s]=dict(orders,**{uid:root["alternative"]});sim.step(*altered)
            else:sim.step(*pair)
        obs=sim.obs(s);update_memory(obs,mem);orders=parent.agent(obs,cfg)
        p=diagnostics(obs,mem,orders,uid,root["target"],cfg)
        row=dict(seed=q['seed'],seat=s,current=next(v for v in options if v['cell']==root['target']),all_root_moves=options,
                 profitable_two_step_clear_alternative=[v["direction"] for v in options if v["distinct_qualified_alternative"] and
                 (v["nearest_enemy"] is None or v["nearest_enemy"]["distance"]>=3)],
                 next_tick=dict(**p,worker_survived_at_target=obs["players"][s][2].get(uid,[None])[0]==root["target"]))
        follow=HERE/f"seat{s}/follow.json"
        if follow.exists() and sr.load(follow)["parent"]!=orders:raise ValueError("native callback differs")
        rows.append(row)
    sr.retain(HERE/'abstention_diagnostics.json',rows)
    for r in rows:print(r["seat"],r["current"]["nearest_enemy"],r["next_tick"]["nearest_enemy"],r["next_tick"]["reasons"],r["next_tick"]["current_view_observed_sum"],r["next_tick"]["net_ground"],r["profitable_two_step_clear_alternative"])
if __name__=="__main__":audit()

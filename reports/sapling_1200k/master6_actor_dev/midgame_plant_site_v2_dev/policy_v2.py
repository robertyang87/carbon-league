"""Propagate own-public lower bounds for a two-turn planting trip on the torus."""
import numpy as np


def next_floor(observation, ground_floor, cfg, orders=None):
    seat = observation["player"]
    limit = ground_floor.copy()
    adjacent_trees = np.zeros(len(limit))
    for owner, side in enumerate(observation["players"]):
        for cell, age in side[3].values():
            limit[cell] = 0.
            adjacent_trees[_TO[cell][1:]] += 1
        for cell in side[1].values():
            limit[cell] = 0.
        for worker, (cell, cargo, role) in side[2].items():
            if owner == seat and orders is not None and orders.get(worker) in _DIRECTION[1:]:
                continue
            if role == "PLANTER":
                limit[cell] = 0.
            else:
                limit[cell] *= max(0., 1. - cfg["initialCollectRate"])
    return np.maximum(0., limit * np.maximum(0., 1. - adjacent_trees * cfg["cellAbsorptionRate"]) - .00051)


def record_floor(obs, mem, cfg):
    tick=obs['step']
    if mem.get("floor_step")==tick:
        return
    current = mem.get("following_floor", np.zeros(len(obs['carbon']))).copy()
    if "floor_step" in mem and mem["floor_step"]+1!=tick:
        raise ValueError("skipped public turn in floor history")
    update_memory(obs,mem)
    for cell, value in enumerate(obs['carbon']):
        known=mem.get(cell)
        if known is not None and known[1]==tick:current[cell]=value
    mem["current_floor"]=current
    mem["following_floor"]=next_floor(obs,current,cfg)
    mem["floor_step"]=tick


def choose(obs,mem,joint,cfg,weights):
    t=obs['step'];s=obs['player'];own,enemy=obs['players'][s],obs['players'][1-s]
    if not 100<=t<200:return None
    enemytrees={p for p,age in enemy[3].values()}
    fee=quote(obs,cfg)
    floors=next_floor(obs,mem['current_floor'],cfg,joint)
    best=None;best_key=None
    for uid,(cell,_,role) in own[2].items():
        old=joint.get(uid)
        if role!='PLANTER' or old not in _DIRECTION[1:]:continue
        prior=int(_TO[cell,_DIRECTION.index(old)])
        if cell in enemytrees or prior in enemytrees:continue
        original=values(obs,mem,uid,prior,cfg)
        net0=max(0.,predicted(original,weights)-fee) if original is not None else 0.
        for idx,raw_cell in enumerate(_TO[cell][1:],1):
            target=int(raw_cell)
            if target==prior or not mode(obs,joint,uid,target,cfg):continue
            nearest=min((int(_D[target,w[0]]) for w in enemy[2].values()),default=99)
            lower=float(floors[_TO[target][1:]].sum())
            if nearest<3 or lower<120:continue
            f=values(obs,mem,uid,target,cfg)
            if f is None or f[3]<.5 or f[2]<1.2:continue
            net=predicted(f,weights)-fee; gain=net-net0
            if gain<=10:continue
            key=gain,uid,-idx
            if best_key is None or key>best_key:
                best_key=key
                best=dict(t=t,worker=uid,old=old,alternative=_DIRECTION[idx],tree=None,cell=cell,target=target,
                          prior_destination=prior,known_fraction=f[3],remembered_total=f[2]*100,quoted_fee=fee,
                          target_net=net,alternative_prior_net=net0,predicted_advantage=gain,
                          guaranteed_next_soil=lower,minimum_opponent_distance=nearest)
    return best

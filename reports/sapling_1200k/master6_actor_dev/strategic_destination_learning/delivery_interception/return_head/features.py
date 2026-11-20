"""Deployable pre-action geometry and public history for one enemy collector."""
import importlib.util
from pathlib import Path
import math

import numpy as np

HERE = Path(__file__).resolve().parent
FREIGHT_PATH = HERE.parents[2] / "freight_horizon/predictor.py"
spec = importlib.util.spec_from_file_location("return_public_freight_features", FREIGHT_PATH)
freight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freight)
EXTRA = (
    "enemy_log_cash", "own_log_cash", "enemy_total_log_cargo", "own_total_log_cargo",
    "opponent_collectors", "opponent_planters", "own_collectors", "own_planters",
    "own_tree_count", "enemy_tree_count", "nearest_own_planter", "nearest_enemy_planter",
    "near_base_1_home_distance", "near_base_1_nearest_own_collector", "near_base_1_nearest_own_planter",
    "near_base_1_enemy_workers_within_two", "near_base_1_own_trees_within_two", "near_base_1_enemy_trees_within_two",
    "near_base_2_home_distance", "near_base_2_nearest_own_collector", "near_base_2_nearest_own_planter",
    "near_base_2_enemy_workers_within_two", "near_base_2_own_trees_within_two", "near_base_2_enemy_trees_within_two",
    "previous_seen", "previous_home_progress", "previous_stationary", "previous_cargo_change",
    "ten_seen", "ten_home_progress", "ten_net_displacement", "ten_cargo_change",
)
NAMES = ("log_cargo", *freight.NAMES[1:], *EXTRA)


def vector(history, uid):
    if len(history)!=11:
        raise ValueError("prediction requires eleven public views ending at the current root")
    obs=history[-1]
    side,t=obs["player"],obs["step"]
    if any(o["player"]!=side or o["step"]!=t-10+i or "full_carbon" in o for i,o in enumerate(history)):
        raise ValueError("mixed seats, future observations, gaps or hidden board supplied")
    own = obs["players"][side]
    enemy = obs["players"][1-side]
    p,c,role = enemy[2][uid]
    if role!="COLLECTOR" or c<100:
        raise ValueError("requires the registered loaded opposing collector")
    dist = freight.DIST
    homes = sorted(enemy[1].values(), key=lambda b:(dist[p][b],b))
    if len(homes)!=2 or not 2<=dist[p][homes[0]]<=10:
        raise ValueError("wrong number of warehouses or outside registered alert window")
    row = list(freight.public_features(obs,history[:-1])[uid])
    row[0] = math.log1p(c)
    oc = [v[0] for v in own[2].values() if v[2]=="COLLECTOR"]
    op = [v[0] for v in own[2].values() if v[2]=="PLANTER"]
    ep = [v[0] for v in enemy[2].values() if v[2]=="PLANTER"]
    ot = [v[0] for v in own[3].values()]
    et = [v[0] for v in enemy[3].values()]
    nearest = lambda a,positions: min((dist[a][v] for v in positions),default=15)
    row.extend((math.log1p(max(enemy[0],0)),math.log1p(max(own[0],0)),
                math.log1p(sum(v[1] for v in enemy[2].values())),
                math.log1p(sum(v[1] for v in own[2].values())),
                len(enemy[2])-len(ep),len(ep),len(oc),len(op),len(ot),len(et),nearest(p,op),nearest(p,ep)))
    for home in homes:
        row.extend((dist[p][home],nearest(home,oc),nearest(home,op),
                    sum(dist[home][v[0]]<=2 for v in enemy[2].values()),
                    sum(dist[home][pos]<=2 for pos in ot),sum(dist[home][pos]<=2 for pos in et)))
    for lag in (1,10):
        past=history[-1-lag]["players"][1-side][2].get(uid)
        if past is None:
            row.extend((0,0,0,0))
            continue
        jump=dist[p][past[0]]
        row.extend((1,nearest(past[0],homes)-nearest(p,homes),int(jump==0) if lag==1 else jump,c-past[1]))
    values=np.asarray(row,dtype=np.float64)
    if len(values)!=len(NAMES) or not np.isfinite(values).all():
        raise ValueError("incomplete or nonfinite current-public features")
    return values

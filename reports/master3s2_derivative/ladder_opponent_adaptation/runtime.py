"""Public root-only action features for enemy workers and eligible recruit sites."""
import numpy as np


def opponent_context(players, enemy, step, move, distance):
    wallet, bases, workers, trees = players[enemy]
    own = players[1-enemy]
    depots = list(bases.values())
    friendly_trees = {v[0] for v in trees.values()}
    enemy_trees = {v[0] for v in own[3].values()}
    all_trees = friendly_trees | enemy_trees
    all_bases = set(depots) | set(own[1].values())
    banks = distance[:,depots].min(1)
    cautious = [u for u in own[2].values() if u[2] == "COLLECTOR"]
    rivals = list(own[1].values()) + [u[0] for u in cautious]
    cols = sum(u[2]=="COLLECTOR" for u in workers.values())
    counts = len(workers)
    results = []
    for unit,(pos,cargo,kind) in workers.items():
        where = move[pos]
        home = banks[pos] - banks[where]
        load = min(2.,cargo/150.)
        threatened = [u[0] for u in cautious if u[1] < cargo]
        risk = np.min(distance[where][:,threatened],1) if threatened else np.full(5,15)
        rival_tree = np.array([int(p in enemy_trees) for p in where],np.float32)
        near_tree = len(all_trees)>0 and min(distance[pos,list(all_trees)])<=2
        pressured = len(rivals)>0 and min(distance[pos,rivals])<=3
        f = np.zeros((5,7),np.float32)
        f[0,0] = 1.
        if kind == "COLLECTOR":
            role = 0
            f[:,1] = home*load
            f[:,2] = (banks[where]==0)*load
            f[:,3] = (risk<=1)*load
            f[:,4] = rival_tree
            f[0,5] = min(1.,counts/10)
            f[0,6] = min(1.,step/180.)
            focus = cargo>=100 and (banks[pos]<=3 or pressured)
        else:
            role = 1
            f[:,1] = rival_tree
            f[:,2] = [int(p in friendly_trees) for p in where]
            f[0,3] = int(pos in enemy_trees)
            f[0,4] = int(pos not in all_trees and pos not in all_bases)
            f[0,5] = min(1.,wallet/150)
            f[0,6] = min(1.,step/180.)
            focus = near_tree
        results.append((role,unit,pos,f,bool(focus)))
    for unit,pos in bases.items():
        f=np.zeros((3,7),np.float32)
        for i in (1,2):
            f[i,0] = 1 if i == 1 else -1
            f[i,1] = 1
            f[i,2] = counts/10
            f[i,3] = (1 if i==1 else -1)*(cols/max(1,counts))
            f[i,4] = min(1.,wallet/150)
            f[i,5] = min(1.,step/180.)
            f[i,6] = int(any(u[0]==pos for u in workers.values()))
        results.append((2,unit,pos,f,bool(wallet>=30 and counts<10)))
    return results


def opponent_adjust(lw,lb,context,coeff):
    for role,unit,pos,f,focus in context:
        v = (lw if role < 2 else lb)[0,pos]
        c = coeff[role]
        v *= c[0]
        v += 1.5*np.tanh((f@c[1:])/1.5)
    return lw,lb

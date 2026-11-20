"""Two collector-only offsets at V2's opponent-root site (after its learned readout)."""
import numpy as np


def features(players, enemy, move_table, distance):
    workers = [(unit[0],unit[1]) for unit in players[enemy][2].values() if unit[2]=="COLLECTOR"]
    positions = np.asarray([p for p,_ in workers],dtype=np.intp)
    cargo = np.asarray([c for _,c in workers],dtype=np.float32)
    phi = np.zeros((len(positions),5,2),dtype=np.float32)
    phi[:,0,0] = 1.
    bases = list(players[enemy][1].values())
    if len(positions) and bases:
        closest = np.min(distance[:,bases],axis=1)
        change = closest[positions,None] - closest[move_table[positions]]
        phi[:,:,1] = np.minimum(cargo/200.,1.)[:,None]*change
    return positions,phi


def adjust(lwo, players, enemy, coefficients, move_table, distance):
    positions, phi = features(players,enemy,move_table,distance)
    lwo[0,positions] += phi @ np.asarray(coefficients,dtype=np.float32)
    return lwo

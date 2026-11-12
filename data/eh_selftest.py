import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'env_src', 'carbon_challenge-master'))
import eval_v53, cnet2_search as S, cnet2_feat as F
from fastsim import FastCarbon
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
CONF = {'MODE': 'mcts', 'CLOCK': 'wall', 'BUDGET': 99.0, 'HARD': 99.0, 'ABORT_DL': None, 'START_DL': None, 'ROOT_S': 10, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'MIX': True, 'MINC': 20.0, 'GUARD': True, 'GUARD_NARROW': True, 'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'STYLES': (0, 1, 2, 3, 4, 5, 6, 7), 'ID_PRIOR': 0.0, 'THREAT_FIRST': True, 'THREAT_MIN': 100.0, 'EH_STEP': 270, 'EH_HEAVY': 300.0, 'EH_HEAVY_SLACK': 4}
for seed in (992001, 992002, 997003):
    side = seed % 2; random.seed(eval_v53.rng_seed(seed, side)); sim = FastCarbon(eval_v53.CFG).reset(seed=seed)
    ag = S.make_search_agent(dict(CONF), weights=os.path.join(W, 'cnet2_b128v9.npz'), rweights=os.path.join(W, 'cnet2_s64v8.npz'))
    opp = eval_v53.load_agent_fn(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'exp', 'submission_cnet2_v13.py'))
    hist = {}
    for step in range(1, 301):
        o = [sim.obs(0), sim.obs(1)]
        a_me = ag(o[side], eval_v53.CFG); a_op = opp(o[1 - side], eval_v53.CFG)
        hist[o[side]['step']] = o[side]
        sim.step(*((a_me, a_op) if side == 0 else (a_op, a_me)))
    final = sim.obs(side); last = hist[max(hist)]
    forced = sorted(set(u for _, u in ag.stats.get('eh_list', [])))
    left_cargo = {u: (last['players'][side][2].get(u, [None, 'gone'])[1]) for u in forced}
    # self-collision: our worker vanishing in the last 15 steps with no enemy worker adjacent beforehand
    selfc = 0
    for t in range(285, max(hist)):
        if t not in hist or t + 1 not in hist: continue
        P0, P1 = hist[t]['players'], hist[t + 1]['players']
        for u, w in P0[side][2].items():
            if u not in P1[side][2] and w[2] == 'COLLECTOR':
                near = [e for e in P0[1 - side][2].values() if F.DIST[w[0], e[0]] <= 2]
                if not near: selfc += 1
    print(seed, 'forced cars', len(forced), 'cargo of forced cars at the last observation', left_cargo, '| own collectors vanishing with no enemy within 2:', selfc,
          '| stranded at end', round(sum(w[1] for w in last['players'][side][2].values())))
# --- detail: trajectory of forced cars in the last game run (seed loop above leaves ag/hist/side of the last seed)

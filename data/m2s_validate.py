import sys, os, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import eval_v53
from concurrent.futures import ProcessPoolExecutor
if __name__ == '__main__':
    ts = [('m2sval', 'exp/cnet2_m2sstand.py', 'exp/submission_cnet2_v14.py', s, s % 2) for s in range(996500, 996512)]
    with ProcessPoolExecutor(4) as ex:
        R = list(ex.map(eval_v53.play_game, ts))
    m = lambda f: round(st.mean(f(r) for r in R), 1)
    print('stand-in vs v14: margin', m(lambda r: r['margin']), 'wins', sum(r['margin'] > 0 for r in R), '/', len(R))
    print('death_cargo stand-in', m(lambda r: r['ledger_pol']['death_cargo']), 'v14', m(lambda r: r['ledger_opp']['death_cargo']))
    print('deposits stand-in', m(lambda r: r['ledger_pol']['deposits']), 'v14', m(lambda r: r['ledger_opp']['deposits']))
    for k in ('100', '200'):
        print('collectors/planters/trees @', k, 'stand-in', [m(lambda r, i=i: r['comp'][k][r['policy_side']][i]) for i in range(3)],
              'v14', [m(lambda r, i=i: r['comp'][k][1 - r['policy_side']][i]) for i in range(3)])

"""bk_check A.py B.py TAG_A_LIVE  -- replay every game of a bigeval live file with margin < -1000 and report bankruptcy"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../code')
import eval_v53
from concurrent.futures import ProcessPoolExecutor
def run(t): return eval_v53.play_game(t)
if __name__ == '__main__':
    a, b, tag = sys.argv[1:4]
    rows = [json.loads(l) for l in open(f'../data/bigeval_live/{tag}.jsonl')]
    bad = [r for r in rows if r['margin'] < -1000]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(run, [(tag + '_bk', a, b, r['seed'], r['seed'] % 2) for r in bad]))
    for r0, r in zip(bad, res):
        print(tag, r['seed'], 'margin', r0['margin'], 'replay', round(r['margin']), 'bankrupt_steps', r['bankrupt_steps'], 'death_cargo', round(r['ledger_pol']['death_cargo']))

"""Independently reconcile the largest wrong-sign predictions with cash postings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from data import HERE,read_game,splits
sys.path.insert(0,str(HERE.parent))
from turning_point_ledger import FastCarbon,CashEvents


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',required=True,type=Path);p.add_argument('--new',required=True,type=Path);a=p.parse_args()
    scores=json.loads((HERE/'scores.json').read_text());parts=splits();result={}
    for split,folder in (('old_val',a.old),('new_diag',a.new)):
        index={g['id']:g for g in parts[split]};result[split]=[]
        for e in scores[split]['counterexamples']:
            game=read_game(folder,index[e['id']]);t=e['t'];s=e['seat']
            frames=game['steps'];sim=FastCarbon(game['configuration']).load_observation(frames[0][0]['observation'])
            sim.cash=CashEvents(sim.cash);totals=[Counter(),Counter()]
            for k in range(t+30):
                sim.cash.events=[]
                sim.step(*[row['action'] or {} for row in frames[k+1]])
                players=frames[k+1][0]['observation']['players']
                if sim._players_obs()!=players:raise ValueError('cash/board replay mismatch')
                if k>=t:
                    for kind,side,cash,*extra in sim.cash.events:totals[side][kind]+=cash
            diff={kind:round(totals[s][kind]-totals[1-s][kind],5) for kind in set(totals[0])|set(totals[1])}
            if abs(sum(diff.values())-e['actual'])>1e-4:raise ValueError('cash postings fail to balance')
            result[split].append({**e,'observed_cash_components':dict(sorted(diff.items(), key=lambda kv:-abs(kv[1])))})
    (HERE/'failure_ledgers.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()

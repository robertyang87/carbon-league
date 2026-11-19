"""Audit the two preidentified economic periods without changing the published selection rule."""
import argparse,importlib.util,json,sys
from collections import Counter,defaultdict
from pathlib import Path
from prepare import f,P,HERE,FULL,ROOT,sha
source=FULL/'direct_original/economy/analyze.py'
econ=f.imported('direct_native_previous_cash_account',source)

def case(row):
    s,seat,arm=row['seed'],row['seat'],row['sign']
    dst=HERE/'economy'/f'{arm}_{s}_{seat}.json'
    if dst.exists():
        out=f.read(dst);assert out['source_sha256']==row['raw_sha256'];return out
    path=HERE/'games'/row['driver_label']/f'raw_{s}_{seat}.json.gz';assert sha(path)==row['raw_sha256'];tape=f.read(path)
    board=tape['initial_terrain_u8_sha256'];cards=dict(path=str(path.relative_to(ROOT)),seed=s,seat=seat,terrain=board)
    ledger,_,_,_,_=econ.process(cards,events=False); phases=ledger['phases']
    other='plus' if arm=='minus' else 'minus'; agent=f.loaded(HERE/(other+'.py'))
    f.W.check_seed=lambda seed: board if seed==s else (_ for _ in ()).throw(ValueError('seed'))
    w=f.W.World(s);assert w.terrain_sha256==board
    counts=defaultdict(Counter);example=[]
    for t,joint in enumerate(tape['actions']):
        obs=w.obs(seat)
        if t==0 or t>=100:
            counterfactual=agent.agent(obs,w.config)
            if t>=100:
                c=counts[100 if t<200 else 200];executed=joint[seat];c['roots']+=1;c['joint_different']+=executed!=counterfactual
                if executed!=counterfactual:
                    detail={}
                    for uid in sorted(executed.keys()|counterfactual.keys()):
                        if executed.get(uid)==counterfactual.get(uid):continue
                        role='base' if uid in obs['players'][seat][1] else obs['players'][seat][2][uid][2].lower()
                        c[role]+=1;detail[uid]=[role,executed.get(uid,'NONE'),counterfactual.get(uid,'NONE')]
                    example.append(dict(t=t,changed_units=detail))
        else:agent._FEAT.observe(obs)
        w.advance(joint)
    late=[phases['200'][s0].get('own_delivery',0.) for s0 in (0,1)]
    mid=[phases['100'][s0].get('tree_ground',0.) for s0 in (0,1)]
    out=dict(sign=arm,seed=s,seat=seat,source_sha256=sha(path),ledger_sha256=sha(source),period_actions={k:dict(v) for k,v in counts.items()},joint_witnesses=example,
        mid_tree_ground_own_original=mid,late_delivery_own_original=late,official_score_own_original=ledger['scores_own_enemy'])
    f.put(dst,out);return out
def main():
    rows=f.read(HERE/'scores.json');cases=[case(r) for r in rows];diff=[]
    for s in P['seeds']:
        for seat in (0,1):
            out={'seed':s,'seat':seat}
            a,b=next(v for v in cases if (v['seed'],v['seat'],v['sign'])==(s,seat,'plus')),next(v for v in cases if (v['seed'],v['seat'],v['sign'])==(s,seat,'minus'))
            for metric in ('mid_tree_ground_own_original','late_delivery_own_original'):
                v,w=a[metric],b[metric]
                out[metric]=dict(plus=v,minus=w,own_delta_plus_minus=v[0]-w[0],opponent_delta_plus_minus=v[1]-w[1],relative_delta_plus_minus=(v[0]-v[1])-(w[0]-w[1]))
            diff.append(out)
    f.put(HERE/'economic_comparison.json',dict(selection=sha(HERE/'selection.json'),program=sha(Path(__file__)),case_files=[f'economy/{v["sign"]}_{v["seed"]}_{v["seat"]}.json' for v in cases],pairs=diff,
        note='Paired maps and seats; original naturally samples separate trajectories per game. Actual ledger differences are descriptive. The alternative command is also independently decoded on each exact occupied state and does not alter the recorded game. Neither metric enters the registered decision.'))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path);args=parser.parse_args()
    if args.case:case(f.read(args.case))
    else:main()

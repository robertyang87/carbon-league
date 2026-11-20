"""For nonterminal roots, isolate one-step income and worker losses under the actually observed examiner reply."""
from collections import Counter
import copy,gzip,json,zipfile
import numpy as np
from panel import PLAN,HERE,FAMILY,sha,load,sources,put
GAME=FAMILY.parents[1]/'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original/economy.py'
econ=load('immediate_carbon',GAME)

def run():
    rows=[];cfg=econ.config()
    for card in PLAN['tickets']:
        side=card['seat'];seed=card['seed']
        _,turns,frames=sources(card)
        panel=json.loads(gzip.decompress((HERE/'roots'/f'{seed}_{side}.json.gz').read_bytes()))['rows']
        selected={r['step']:r for r in panel if r['step']<298}
        sim=econ.FastCarbon(cfg).reset(seed=seed);sim.cash=econ.CashEvents(sim.cash)
        assert sha(bytes(map(int,sim.carbon)))==card['terrain']
        for step,line in enumerate(turns):
            players=frames[step]['observations'][side]['players'];assert sim.obs(side)['players']==players
            actions=[c['response']['action'] for c in line['calls']]
            if step in selected:
                r=selected[step];assert r['real_examiner_root']==actions[1-side]
                out=dict(seed=seed,seat=side,step=step,original_enemy=actions[1-side],changed=r['M1']['issued']!=r['M2']['issued'])
                for name in ('M1','M2'):
                    b=copy.deepcopy(sim);b.cash.events.clear()
                    joint=list(actions);joint[side]=r[name]['issued'];b.step(*joint)
                    flow=econ.cash_flow(b.cash.events)
                    result=b.obs(side)['players']
                    out[name]=dict(cash=list(b.cash),own_minus_opponent=round(b.cash[side]-b.cash[1-side],2),
                                   deltas_by_seat=flow,lost_workers=[[u for u in p[2] if u not in result[s][2]] for s,p in enumerate(players)])
                out['one_step_margin_difference']=round(out['M2']['own_minus_opponent']-out['M1']['own_minus_opponent'],2)
                rows.append(out)
            sim.cash.events.clear();sim.step(*actions)
        with zipfile.ZipFile(FAMILY/card['source']) as archive:
            official=json.loads(archive.read('parent/receipt.json'))
        assert sim.step_count==official['official_transitions']==299
        assert list(sim.cash)==official['official_scores_by_seat']
    nums=[r['one_step_margin_difference'] for r in rows]
    result=dict(diagnostic='One observed simultaneous original action and one verified engine advance; no independent continuation, terminal outcomes or causal claim about later finish.',rows=rows,
                n=len(rows),positive=sum(x>0 for x in nums),same=sum(x==0 for x in nums),negative=sum(x<0 for x in nums),
                mean=float(np.mean(nums)),per_map={s:float(np.mean([r['one_step_margin_difference'] for r in rows if r['seed']==s])) for s in sorted({r['seed'] for r in rows})})
    put(HERE/'one_step.json.gz',result)
    print({k:v for k,v in result.items() if k not in ('rows','diagnostic')})
if __name__=='__main__':run()

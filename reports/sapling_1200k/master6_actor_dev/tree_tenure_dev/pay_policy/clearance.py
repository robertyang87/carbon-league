"""Inspect the contemporaneous opponent orders after the pre-action selection, never as live inputs."""
from collections import Counter
import copy
import gzip
import json
from opportunities import D, ROOT, CashEvents, FastCarbon, read, sha
from resolution import both


def main():
    source=D/'opportunities.json.gz'; doc=read(source); rows=[]
    for g in doc['games']:
        raw=read(ROOT/g['source']);sim=FastCarbon(raw['game_configuration']).reset(seed=g['seed']); s=g['seat']
        cases={r['t']:r for r in g['selected']}
        for t,orders in enumerate(raw['actions']):
            if t in cases:
                tr=cases[t]; uid=tr['uid']; cell=tr['cell']; budget=tr['own_cash']-tr['hire_reserve']-tr['other_planter_reserve']
                comparison=both(sim,orders,s,tr)
                outcome=comparison['pay'];endpoints=outcome['other_endpoints']
                survived=outcome['alive_after'];spend=outcome['focus_debit']
                assert abs(spend-tr['forced_step_comparison']['worker_buy_cost'])<1e-6
                assert abs(comparison['parent']['focus_debit']-tr['actual_parent']['worker_buy_cost'])<1e-6
                if spend>0: verdict='charged'
                elif not survived and any(e['side']!=s for e in endpoints):verdict='absent_after_shared_enemy_destination'
                elif not survived:verdict='absent_other'
                else:verdict='survived_no_charge'
                rows.append(dict(seed=g['seed'],s=s,t=t,uid=uid,window=tr['window'],kind=tr['kind'],changed=tr['changed'],alive_after=survived,
                    own_residual_budget=budget,price=tr['quoted_cost'],paid=spend,classification=verdict,other_target_endpoints=endpoints,
                    same_order_parent=comparison['parent'],survival_flip=comparison['survival_flip'],nearest_enemy_before=outcome['nearest_enemy_before']))
            sim.step(*orders)
    assert len(rows)==doc['selected']
    out=dict(source_sha256=sha(source),total=len(rows),changed=dict(Counter(r['classification'] for r in rows if r['changed'])),
             changed_survival=dict(Counter(f"parent_{int(r['same_order_parent']['alive_after'])}_pay_{int(r['alive_after'])}" for r in rows if r['changed'])),
             changed_affordability_by_reach={k:dict(Counter(r['classification'] for r in rows if r['changed'] and ((r['nearest_enemy_before'] is not None and r['nearest_enemy_before']<=1)==b))) for k,b in [('reachable',True),('unreachable',False)]},
             unchanged=dict(Counter(r['classification'] for r in rows if not r['changed'])),cases=rows)
    (D/'clearance.json.gz').write_bytes(gzip.compress(json.dumps(out,separators=(',',':')).encode(),mtime=0));print(out['changed'],out['unchanged'])


if __name__=='__main__':main()

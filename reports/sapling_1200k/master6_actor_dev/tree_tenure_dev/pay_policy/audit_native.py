"""Attach realized cost and first ownership income to each chronological selection on eight actual tapes."""
from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path
from resolution import both
from opportunities import CashEvents, FastCarbon, D, ROOT, pack, read, sha

KEYS = ('own_delivery','opponent_worker_delivery_to_this_base','tree_absorption','tree_collision_cash',
        'base_collision_cash','hire_collector','hire_planter','plant','seize','rounding')


def ledger(m,path,arm):
    r=read(path); s=r['candidate_seat']; cfg=r['game_configuration']
    reg=read(D/'protocol.json'); meta=read(D/'package.json')
    expected=meta['candidate_sha256'] if arm=='candidate' else reg['parent_sha256']
    assert r['pack_sha256']==expected and r['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1'
    assert r['seed'] in reg['new_dev']['seeds'] and not r['errors'] and not r['audit_messages']
    sim=FastCarbon(cfg).reset(seed=r['seed']); initial=list(sim.cash); sim.cash=CashEvents(list(sim.cash))
    total=[defaultdict(float),defaultdict(float)]; windows=[0,0]; selected=[]; active={}
    def close(tid,end,reason):
        tr=active.pop(tid);tr['held_transitions']=end-tr['paid_turn']+(1 if reason=='terminal' else 0)
        tr['end']=reason
    for t,orders in enumerate(r['actions']):
        obs=sim.obs(s);window=0 if 45<=t<=125 else (1 if 150<=t<=235 else None)
        tr=None
        if window is not None:windows[window]+=1
        if t==0 or (window is not None and window not in m._PAY_USED):
            old=0 if t==0 else len(m._PAY_TRACE)
            cand=m.agent(obs,cfg)
            if len(m._PAY_TRACE)>old:
                tr=dict(m._PAY_TRACE[-1]);tr['same_enemy_both_actions']=both(sim,orders,s,tr);tr['actual_changed_by_policy']=arm=='candidate' and tr['changed']
                tr['actual_parent_at_root']=tr.pop('base')
                assert orders[s]==(cand if arm=='candidate' else tr['actual_parent_at_root'])
                tr.update(ordered=orders[s].get(tr['uid'],'NONE')=='NONE',debited=0.,gross_absorption=0.,enemy_worker_absorption=0.,
                          collision_cash=0.,held_transitions=None,paid_turn=None,end=None,tree_id=None)
                selected.append(tr)
            else: assert orders[s]==cand
        elif len(m._PAY_USED)<2 and t<=235:
            m._FEAT.observe(obs)
        sim.cash.events.clear(); sim.step(*orders)
        positions={tree.pos:tid for tid,tree in sim.trees.items()}
        for kind,who,val,*args in sim.cash.events:
            assert kind in KEYS
            total[who][kind]+=val
            if kind in ('plant','seize'):
                uid,cell=args;tid=positions[cell]
                if tid in active: close(tid,t+1,'seized')
                if tr is not None and who==s and uid==tr['uid']:
                    assert tr['ordered'] and tr['kind']==kind and tr['cell']==cell and abs(tr['quoted_cost']+val)<.001
                    tr.update(debited=-val,paid_turn=t+1,tree_id=tid); active[tid]=tr
            elif kind in ('tree_absorption','tree_collision_cash'):
                tid=args[0] if kind=='tree_absorption' else args[2]
                if tid in active:
                    assert who==s
                    if kind=='tree_absorption':
                        active[tid]['gross_absorption']+=val;active[tid]['enemy_worker_absorption']+=args[2]
                    else:active[tid]['collision_cash']+=val
        if tr is not None:
            after=sim.obs(s)['players'][s][2].get(tr['uid'])
            tr['actual_survived_next']=after is not None
            tr['actual_cell_next']=after[0] if after is not None else None
            comp=tr['same_enemy_both_actions']['pay' if arm=='candidate' else 'parent']
            assert tr['actual_survived_next']==comp['alive_after'] and abs(tr['debited']-comp['focus_debit'])<1e-6
        for tid in list(active):
            if tid not in sim.trees:close(tid,t+1,'expired')
    for tid in list(active):close(tid,len(r['actions']),'terminal')
    for tr in selected:tr['first_tenure_net']=tr['gross_absorption']+tr['collision_cash']-tr['debited']
    assert all(abs(initial[i]+sum(total[i].values())-sim.cash[i])<1e-4 for i in (0,1))
    assert max(abs(sim.cash[i]-r['last_post_cash'][i]) for i in (0,1))<.011
    assert len(selected)<=2
    return dict(arm=arm,seed=r['seed'],seat=s,terrain=r['initial_terrain_u8_sha256'],raw=str(path.relative_to(ROOT)),raw_sha256=sha(path),
                full_windows=windows,selected=selected,turns=len(r['actions']),cash_events_by_seat=total,
                cash_end=list(sim.cash),official=r['official_score_by_seat'],official_margin=r['margin'],wdl=r['result'],extinct=r['extinct_by_seat'])


def main(paths):
    m=pack(); meta=read(D/'package.json'); rows=[]
    for path in paths:
        r=read(path);arm='candidate' if r['pack_sha256']==meta['candidate_sha256'] else 'parent';rows.append(ledger(m,path,arm))
    grouped={(r['seed'],r['seat'],r['arm']):r for r in rows}
    p=read(D/'protocol.json')
    assert set(grouped)=={(s,n,a) for s in p['new_dev']['seeds'] for n in (0,1) for a in ('candidate','parent')} and len(rows)==8
    paired=[]
    for seed in p['new_dev']['seeds']:
        for s in (0,1):
            c=grouped[seed,s,'candidate']; b=grouped[seed,s,'parent']
            assert c['terrain']==b['terrain']
            delta={}
            for i,label in [(s,'own'),(1-s,'rival')]:
                delta[label]={k:c['cash_events_by_seat'][i].get(k,0)-b['cash_events_by_seat'][i].get(k,0) for k in KEYS}
            paired.append(dict(seed=seed,seat=s,official_own=c['official'][s]-b['official'][s],
                               official_minus_rival=b['official'][1-s]-c['official'][1-s],margin=c['official_margin']-b['official_margin'],cash_delta=delta))
    csel=[x for r in rows if r['arm']=='candidate' for x in r['selected']]
    out=dict(protocol_sha256=sha(D/'protocol.json'),package_sha256=meta['candidate_sha256'],games=rows,paired=paired,summary=dict(
        games_per_arm=4,wins={arm:sum(r['wdl']=='win' for r in rows if r['arm']==arm) for arm in ('candidate','parent')},
        mean_margin_difference=sum(d['margin'] for d in paired)/4,mean_own_difference=sum(d['official_own'] for d in paired)/4,
        mean_minus_rival_difference=sum(d['official_minus_rival'] for d in paired)/4,selected=len(csel),changed=sum(x['changed'] for x in csel),
        charged_after_changed=sum(x['debited']>0 for x in csel if x['changed']),
        root_cost_after_changed=sum(x['debited'] for x in csel if x['changed']),root_income_after_changed=sum(x['gross_absorption']+x['collision_cash'] for x in csel if x['changed'])))
    (D/'game_economy.json.gz').write_bytes(gzip.compress(json.dumps(out,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    print(json.dumps(out['summary']),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('raw',type=Path,nargs=8)
    main([p.resolve() for p in a.parse_args().raw])

"""Separate the choice of scenario aggregation, adaptive work, and the independent observed root reply."""
from collections import Counter
from pathlib import Path
import gzip,json
import numpy as np
from panel import HERE, PLAN, LOCK, put


def pmse(sampled, actual, ids):
    if not sampled: return []
    out=[]
    for u in ids:
        probs=Counter(s.get(u,'NONE') for s in sampled)
        expected=actual.get(u,'NONE')
        out.append([probs[expected]/len(sampled),sum((count/len(sampled))**2 for count in probs.values())-2*probs[expected]/len(sampled)+1])
    return out


def summary(data):
    n=len(data)
    two=lambda a:a[1]
    metrics={}
    def stat(name,values):
        nums=np.asarray(values,dtype=float)
        metrics[name]={'n':len(values),'mean':float(nums.mean()) if len(nums) else None,'sum':float(nums.sum())}
    def h(a):return a['clock']['H']
    pairs=[(r['M1'],r['M2']) for r in data]
    stat('issued_change',[a['issued']!=b['issued'] for a,b in pairs])
    stat('A0_changed_M1',[a['issued']!=a['menu'][0] for a,b in pairs])
    stat('A0_changed_M2',[b['issued']!=b['menu'][0] for a,b in pairs])
    for tag in ('M1','M2'):
        for what in ('K','H','wall','cpu'):
            vals=[r[tag]['clock'][what] for r in data]
            stat(tag+'_'+what,vals)
            if what in ('K','H'):metrics[tag+'_'+what]['distribution']=dict(Counter(vals))
            if what=='wall':metrics[tag+'_'+what].update(p95=float(np.percentile(vals,95)),max=max(vals),slow_gt1=sum(v>1 for v in vals))
        stat(tag+'_menu_physical_finals',[len(r[tag]['alive']) for r in data])
    stat('same_reported_depth',[h(a)==h(b) for a,b in pairs])
    stat('same_complete_physical_menu',[a['menu']==b['menu'] for a,b in pairs])
    stat('M1_issued_missing_from_M2_menu',[a['issued'] not in b['menu'] for a,b in pairs])
    stat('M2_issued_missing_from_M1_menu',[b['issued'] not in a['menu'] for a,b in pairs])
    changed=[(a,b) for a,b in pairs if a['issued']!=b['issued']]
    stat('changed_with_M1_absent_M2_menu',[a['issued'] not in b['menu'] for a,b in changed])
    stat('changed_despite_same_menu',[a['menu']==b['menu'] for a,b in changed])
    available=[b for a,b in pairs if b['individual_winners']]
    stat('within_M2_second_draw_changes_choice_over_draw0',[b['chosen']!=b['individual_winners'][0] for b in available])
    stat('within_M2_draw_winners_disagree',[b['individual_winners'][0]!=b['individual_winners'][1] for b in available])
    eligible=[r for r in data if len(r['M1']['sample_enemy'])==1 and len(r['M2']['sample_enemy'])==2]
    stat('enemy_two_samples_disagree',[r['M2']['sample_enemy']['0']!=r['M2']['sample_enemy']['1'] for r in eligible])
    for nsc,tag in ((1,'M1'),(2,'M2')):
        matches=[];brier=[];density=[]
        for r in eligible:
            real=r['real_examiner_root'];ss=list(r[tag]['sample_enemy'].values())
            units=pmse(ss,real,r['enemy_entities'])
            if not units:continue
            frame=np.mean(units,axis=0)
            matches.append(frame[0]);brier.append(frame[1]);density.append(sum(a==real for a in ss)/nsc)
        stat(tag+'_sampled_entity_accuracy_root_equal_weight',matches)
        stat(tag+'_entity_categorical_brier_root_equal_weight',brier)
        stat(tag+'_sampled_probability_of_actual_joint_order',density)
    return metrics


def main():
    data=[]
    for t in PLAN['tickets']:
        d=json.loads(gzip.decompress((HERE/'roots'/f"{t['seed']}_{t['seat']}.json.gz").read_bytes()))
        assert d['protocol']==LOCK and len(d['rows'])==len(PLAN['steps'])
        for row in d['rows']:
            data.append(dict(row,seed=t['seed'],seat=t['seat']))
    out=dict(protocol=LOCK,n_roots=len(data),maps=len({r['seed'] for r in data}),
             all=summary(data),per_map={s:summary([r for r in data if r['seed']==s]) for s in sorted({r['seed'] for r in data})},
             step_summary={s:summary([r for r in data if r['step']==s]) for s in PLAN['steps']})
    put(HERE/'panel_summary.json',out)
    for k,v in out['all'].items():print(k,v)
if __name__=='__main__':main()

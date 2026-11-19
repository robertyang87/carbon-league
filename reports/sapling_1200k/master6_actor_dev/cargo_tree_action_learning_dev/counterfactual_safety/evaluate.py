"""Map-held regular-entry comparison and one-step opponent-response assurance. No new native examiner calls."""
import collections, gzip
import numpy as np
from gather import HERE,EXP,CFG,read,sha,arena,FastCarbon,outcome,policy,write

def aggregate(results, source):
    accum=collections.defaultdict(list)
    for r in results:
        for arm,v in r[source].items():
            for key in ((r['split'],r['tag'],arm),(r['split'],r['tag'],str(r['seed']),arm),(r['split'],r['tag'],'seat'+str(r['own_seat']),arm)):
                accum[key].append((r,v))
    output={}
    for key,items in accum.items():
        vals=[v for _,v in items];rows=[r for r,_ in items]
        hit=[v['action']!=row[source]['parent']['action'] for row,v in items]
        output['/'.join(key)]=dict(roots=len(items),maps=len(set(r['seed'] for r in rows)),changed_tagged=sum(hit),
                                     changed_team=sum(v.get('team_changed',0) for v in vals),
                                     changed_other_same_tag=sum(int(v.get('team_changed',0))*(not h) for v,h in zip(vals,hit)),
                                     unchanged_tagged=len(items)-sum(hit),survived=sum(v['alive'] for v in vals),
                                     immediate_deposits=sum(v['bank']>0 for v in vals),can_deposit_in_one=sum(r['any_bank'] for r,v in items),
                                     stayed=sum(v['action'] in (0,'STAY') for v in vals),stayed_alive=sum(v['action'] in (0,'STAY') and v['alive'] for v in vals),
                                     retained_harvest_gain=sum(v['held']+v['bank']-r['cargo'] for r,v in items if r['tag']=='harvest'),
                                     dead=sum(not v['alive'] for v in vals),avoidable_dead=sum(not v['alive'] and r['has_surviving_option'] for r,v in items),
                                     actual_tagged_delivery=float(sum(v['bank'] for v in vals)),retained_plus_deposit=float(sum(v['bank']+v['held'] for v in vals)),
                                     paid_trees=sum(int(v['bought']) for v in vals))
    return output

def main():
    f=read(HERE/'fit.json');rows=read(HERE/'roots.json.gz')
    with np.load(HERE/'pairs.npz') as x:base=x['logits'][:,1];mask=x['mask'];label=x['labels']
    with np.load(HERE/'crossfitted.npz') as x:student={'parent':base,**{a:x[a+'_student'] for a in ('control','candidate')}}
    arms=('parent','control','candidate');sources={'parent':EXP/'candidate.py',**{a:HERE/(a+'.py') for a in arms[1:]}}
    models={arm:policy.load(path) for arm,path in sources.items()}
    for arm in arms[1:]:assert sha(sources[arm])==f['outputs'][arm]['pack']
    query=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']!='train':query[r['raw'],r['t']].append((i,r))
    allrows=[];panel=read(EXP/'panel.json')
    for card in panel['cards']:
        if panel['split'][card['terrain']]=='train':continue
        raw=read(arena.ROOT/card['path']);s=card['seat'];cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=card['seed']);fork=FastCarbon(cfg)
        for t,joint in enumerate(raw['actions']):
            obs=sim.obs(s);queries=query[card['sha256'],t]
            if t==0 or queries:
                physical={a:m.agent(obs,cfg) for a,m in models.items()};snap=sim.get_state()
                for i,r in queries:
                    uid=r['uid'];slot=r['slot'];feat=models['parent']._FEAT;parent=feat.labels(physical['parent']);assert parent[slot]==r['parent']
                    regular={};own_network={}
                    for a in arms:
                        joint_arm=[dict(q) for q in joint];joint_arm[s]=physical[a]
                        alive,held,bank,bought=outcome(fork,snap,s,uid,joint_arm)
                        regular[a]=dict(action=physical[a].get(uid,'STAY'),alive=bool(alive),held=held,bank=bank,bought=bool(bought),team_changed=physical[a]!=physical['parent'])
                        alt=int(np.where(mask[i],student[a][i],-np.inf).argmax())
                        aa,hh,pp,bb=label[i,alt].tolist()
                        own_network[a]=dict(action=alt,alive=bool(aa),held=hh,bank=pp,bought=bool(bb))
                    assert np.allclose([float(regular['parent'][k]) for k in ('alive','held','bank','bought')],label[i,parent[slot]],rtol=1e-5,atol=1e-4)
                    allrows.append(dict(index=i,**{k:r[k] for k in ('split','tag','terrain','seed','own_seat','t','uid','raw')},
                                         cargo=r['cargo'],any_bank=bool((label[i,mask[i],2]>0).any()),has_surviving_option=bool(label[i,mask[i],0].any()),regular=regular,standalone_on_parent_prefix=own_network))
            else:
                for m in models.values():m._FEAT.observe(obs)
            sim.step(*joint)
        print(card['seed'],s,len(allrows),flush=True)
    assert len(allrows)==sum(r['split']!='train' for r in rows)
    standalone=aggregate(allrows,'standalone_on_parent_prefix');regular=aggregate(allrows,'regular')
    cand='test/threat/candidate'; bank='test/bank/candidate'
    verdict={'threat_brier_lower':f['outputs']['candidate']['teacher']['test/threat']['survival_brier']<f['outputs']['control']['teacher']['test/threat']['survival_brier'],
             'standalone_fewer_avoidable_deaths':all(standalone[cand]['avoidable_dead']<standalone['test/threat/'+a]['avoidable_dead'] for a in ('parent','control')),
             'deployed_threat_no_extra_deaths':all(regular[cand]['dead']<=regular['test/threat/'+a]['dead'] for a in ('parent','control')),
             'deployed_bank_no_less':all(regular[bank]['actual_tagged_delivery']+1e-5>=regular['test/bank/'+a]['actual_tagged_delivery'] for a in ('parent','control'))}
    write(HERE/'held_rows.json.gz',allrows)
    write(HERE/'held.json',dict(evaluation_contract=sha(HERE/'evaluation_contract.json'),fit=sha(HERE/'fit.json'),rows=sha(HERE/'held_rows.json.gz'),regular=regular,standalone_on_parent_prefix=standalone,
                               verdict=verdict,joint_mechanism='PASS' if all(verdict.values()) else 'INCOMPLETE'))
if __name__=='__main__':main()

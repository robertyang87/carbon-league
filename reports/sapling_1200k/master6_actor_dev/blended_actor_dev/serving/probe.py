"""Normal-entry four-policy execution on archived test/validation observations with each arm's own prefix."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
from build import ACTOR,BASE,DEV,HERE,build,check,encode,save,sha

ARM_NAMES=('candidate','control','active','no_sapling')
STEPS=(0,20,50,80,110,140,170,200,230,260,285,297)
CACHE=Path('/workspace/sapling-all-top5-20261119/replays')


def package(path,name,ablated=False):
    spec=importlib.util.spec_from_file_location('sapling_blend_public_'+name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if ablated:m._HYBRID['student_weight']=0.
    feature=sys.modules['cnet2_feat']
    baseline_decode=m._return_decode
    actual=m._mixed_actions
    captured={}
    def decode(x,aux,states,delta):
        y=baseline_decode(x,aux,states,delta)
        captured[bool(delta)]=y.copy()
        return y
    def checked_action(x,aux):
        captured.clear()
        result=actual(x,aux)
        plain=captured[False];adjusted=captured.get(True)
        if adjusted is None:
            state='residual_disabled'
            check(np.array_equal(result,plain),'disabled decoder')
        else:
            protected=(np.arange(feature.OWN)>=10) | (x['tok_kind'][:feature.OWN]==1)
            if np.any(plain[protected]!=adjusted[protected]):state='protected_role_fallback'
            elif np.array_equal(plain,adjusted):state='no_proposed_change'
            elif m._return_conflicts(x,adjusted)-m._return_conflicts(x,plain):state='new_conflict_fallback'
            else:state='applied'
            expected=adjusted if state in ('no_proposed_change','applied') else plain
            check(np.array_equal(result,expected),'public action vs traced return guard')
        prefix=feature.Prefix(x,aux);slots={}
        for i in np.flatnonzero(x['tok_mask'][:feature.OWN]):
            p=int(i);legal=prefix.feats(p)[2];act=int(result[p])
            check(bool(legal[act]),f'illegal command in {name} prefix at {p}')
            uid=m._FEAT.own_uid[p]
            slots[uid]={'slot':p,'role':'base' if p>=10 else 'planter' if int(x['tok_kind'][p]) else 'collector',
                        'legal':[j for j,x in enumerate(legal) if bool(x)],
                        'actual':act,'before_return':int(plain[p]),'proposed':int(adjusted[p]) if adjusted is not None else None}
            prefix.advance(p,act)
        m.audit={'return_route':state,'slots':slots}
        return result
    m._return_decode=decode;m._mixed_actions=checked_action
    return m


def seat_view(step,s):
    shared=step[0]['observation'];own=step[s]['observation']
    return {'player':s,'step':shared['step'],'players':shared['players'],'trees':shared['trees'],'carbon':own['carbon']}


def cleaned(x):
    return {k:v for k,v in (x or {}).items() if v and v!='NONE'}


def intact(doc,card,terrain):
    raw=doc['steps'][0][0]['observation']['full_carbon']
    check(all(type(v) in (int,float) and int(v)==v and 0<=v<=255 for v in raw),'unsigned terrain')
    check(hashlib.sha256(bytes(map(int,raw))).hexdigest()==terrain,'terrain fingerprint')
    check(card['seat'] in (0,1) and len(doc['steps'])>1 and len(raw)==doc['configuration']['size']**2,'invalid replay')
    if 'seed' in card and 'randomSeed' in doc['configuration']:
        check(doc['configuration']['randomSeed']==card['seed'],'seed identity')


def tally(roots):
    counters=Counter(); roles=defaultdict(Counter); route={a:Counter() for a in ARM_NAMES}
    for r in roots:
        counters['roots']+=1
        for a in ARM_NAMES:
            ar=r['arms'][a];route[a][ar['return_route']]+=1
            counters[a+'/expert_joint_agreement']+=int(ar['actual_command']==r['demonstration_command'])
        for other in ('control','active','no_sapling'):
            counters['candidate_vs_'+other+'/changed_joint']+=r['arms']['candidate']['actual_command']!=r['arms'][other]['actual_command']
        for uid,detail in r['slots'].items():
            rr=roles[detail['role']];rr['occupied']+=1;demo=detail['demo_canonical']
            for a in ARM_NAMES:
                q=detail['outputs'][a];rr[a+'/demo_agreement']+=q['actual']==demo
                rr[a+'/demo_legal_own_prefix']+=demo in q['legal']
                rr[a+'/nonforced_own_prefix']+=len(q['legal'])>1
                rr[a+'/demo_agree_among_nonforced']+=len(q['legal'])>1 and q['actual']==demo
            for other in ('control','active','no_sapling'):
                rr['candidate_vs_'+other+'/changed_slots']+=detail['outputs']['candidate']['actual']!=detail['outputs'][other]['actual']
    routes={}
    for a,z in route.items():
        proposed=z['protected_role_fallback']+z['new_conflict_fallback']+z['applied']
        routes[a]={'counts':{k:z[k] for k in ('no_proposed_change','protected_role_fallback','new_conflict_fallback','applied','residual_disabled')},'proposal_roots':proposed,'fallback_fraction_of_proposals':(
            (z['protected_role_fallback']+z['new_conflict_fallback'])/proposed if proposed else None),
            'fallback_fraction_of_all':(z['protected_role_fallback']+z['new_conflict_fallback'])/len(roots) if roots else None}
    return {'counts':dict(counters),'roles':{k:dict(v) for k,v in roles.items()},'return_residual':routes}


def run_game(models,card,requested):
    p=CACHE/(card['id']+'.json.gz');check(sha(p)==card['sha256'],f"replay bytes changed: {card['id']}")
    doc=json.loads(gzip.decompress(p.read_bytes()));s=card['seat'];land=card.get('terrain_sha256',card.get('terrain'))
    intact(doc,card,land)
    frames=doc['steps'];max_action=len(frames)-2;ts={t for t in requested if t<=max_action}
    check(0 in ts,'t=0 normal-entry initialization required')
    out=[]
    for t,frame in enumerate(frames[:max(ts)+1]):
        obs=seat_view(frame,s)
        check(obs['step']==t,'missing pre-action frame')
        if t not in ts:
            for m in models.values():m._FEAT.observe(obs)
            continue
        demo=cleaned(frames[t+1][s]['action']);actual={};details={};arm_record={}
        for a,m in models.items():
            submitted=m.agent(obs,doc['configuration'])
            check(all(uid in obs['players'][s][1] or uid in obs['players'][s][2] for uid in submitted),'foreign order')
            actual[a]=cleaned(submitted);audit=m.audit
            check(m._FEAT.to_actions(m._FEAT.labels(submitted))==actual[a],'physical output roundtrip')
            target=m._FEAT.labels(demo)
            for uid,row in audit['slots'].items():
                i=row['slot']
                check(int(m._FEAT.labels(submitted)[i])==row['actual'],'output action is not chosen label')
                if uid not in details:details[uid]={'slot':i,'role':row['role'],'demo_canonical':int(target[i]),'outputs':{}}
                check(details[uid]['slot']==i and details[uid]['demo_canonical']==int(target[i]),'inconsistent visible slot order')
                details[uid]['outputs'][a]={k:v for k,v in row.items() if k not in ('role','slot')}
            arm_record[a]={'return_route':audit['return_route'],'actual_command':actual[a]}
        check(all(len(r['outputs'])==len(models) for r in details.values()),'differing occupied slot set')
        out.append({'t':t,'visible_hash_sha256':hashlib.sha256(encode(obs)).hexdigest(),
                    'demonstration_command':demo,'arms':arm_record,'slots':details})
    return {'id':card['id'],'terrain':land,'seat':s,'split':card['split'],'opponent':card.get('opponent'),
            'wdl':card.get('wdl'),'source_sha256':card['sha256'],'observed_action_turns':max_action+1,
            'predeclared_steps_past_terminal':sorted(set(requested)-ts),'roots':out}


def summarize(games):
    group=defaultdict(list)
    for g in games:
        for key in ('all','split/'+g['split'],'opponent/'+str(g['opponent']),'wdl/'+str(g['wdl']),'seat/'+str(g['seat']),
                    'split/'+g['split']+'/opponent_'+g.get('registration_vs_training','unspecified'),
                    'terrain/'+g['terrain']):group[key].extend(g['roots'])
    tab={k:tally(rows) for k,rows in sorted(group.items())}
    maps=[v for k,v in tab.items() if k.startswith('terrain/')]
    macro={}
    for a in ARM_NAMES:
        macro[a+'/demo_joint_agreement']=float(np.mean([m['counts'][a+'/expert_joint_agreement']/m['counts']['roots'] for m in maps]))
        for role in ('collector','planter','base'):
            z=[v['roles'][role] for v in maps if v['roles'].get(role,{}).get('occupied')]
            macro[a+'/'+role+'_demo_agreement']=float(np.mean([s[a+'/demo_agreement']/s['occupied'] for s in z])) if z else None
    for r in ('control','active','no_sapling'):
        macro['candidate_vs_'+r+'/changed_joint']=float(np.mean([z['counts']['candidate_vs_'+r+'/changed_joint']/z['counts']['roots'] for z in maps]))
    return tab,macro


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['holdout','val','both','exposed_loss'],default='both')
    args=p.parse_args()
    packages=build();panel_path=DEV/'panel.json';panel=json.loads(panel_path.read_bytes());cards=panel['games']
    protocol=json.loads((DEV/'protocol.json').read_bytes())
    check(sha(panel_path)==protocol['panel_sha256'] and packages['base_pack_sha256']==protocol['partner_pack_sha256']
          and packages['parent_npz_sha256']==protocol['parent_weights_sha256'],'training/serving freeze mismatch')
    known={c['opponent'] for c in cards if c['split']=='train'}
    bymap=defaultdict(set)
    for c in cards:bymap[c.get('terrain_sha256',c.get('terrain'))].add(c['split'])
    check(all(k and len(s)==1 for k,s in bymap.items()),'train/validation/test terrain overlap')
    active={'holdout','val'} if args.split=='both' else {args.split}
    if args.split=='exposed_loss':
        report=ACTOR/'data_loss_receipts.json.gz';data=ACTOR/'data_loss_split.json'
        receipt=json.loads(gzip.decompress(report.read_bytes()))
        check(sha(report)==protocol['diagnostic_previously_exposed_losses_receipt'] and
              sha(data)==receipt['data_split_sha256'],'loss panel seal')
        selected=json.loads(data.read_bytes())['games']
        check(all(c['split']=='diagnostic_seen_loss' and c['wdl']=='L' for c in selected),'diagnostic-loss scope')
    else:selected=[c for c in cards if c['split'] in active]
    selected=sorted(selected,key=lambda c:(c['split'],c['id'],c['seat']))
    check(selected and len({(g['id'],g['seat']) for g in selected})==len(selected),'empty/duplicate panel')
    models={a:package(path,a,ablated=(a=='no_sapling')) for a,path in
            [('candidate',HERE/'candidate.py'),('control',HERE/'control.py'),('active',BASE),('no_sapling',BASE)]}
    result=[]
    for card in selected:
        item=run_game(models,card,STEPS)
        item['registration_vs_training']='seen' if card['opponent'] in known else 'unseen'
        item['other_research_exposure']=card.get('other_research_exposure')
        result.append(item)
        if len(result)%5==0:print('complete replay maps/seats',len(result),'of',len(selected),flush=True)
    tables,macro=summarize(result)
    splits=sorted({x['split'] for x in result})
    by_split={k:summarize([x for x in result if x['split']==k])[1] for k in splits}
    rows_path=HERE/('probe_'+args.split+'_roots.json.gz')
    save(rows_path,gzip.compress(encode(result)+b'\n',compresslevel=6,mtime=0))
    proof={'scope':'Public decisions on demonstrated histories. Each normal entry completes its own legal autoregressive prefix and applies the frozen return correction. Non-sampled frames only advance the actual acting-seat observation history. No live responding game or counterfactual score in this receipt.',
           'ablation':'Same active composite with student_weight=0; expert_weight=.75 and return residual retained, which matters to the scale of the correction.',
           'steps':list(STEPS),'split':args.split,'registered_terrains':len(bymap),'maps':len({r['terrain'] for r in result}),
           'map_seats':len(result),'quantized_packages':packages,'panel_sha256':sha(panel_path),'protocol_sha256':sha(DEV/'protocol.json'),'probe_sha256':sha(Path(__file__)),
           'roots_sha256':sha(rows_path),'macro_terrain_equal':macro,'macro_terrain_equal_per_split':by_split,'tables':tables}
    save(HERE/('probe_'+args.split+'.json'),json.dumps(proof,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n')
    print('root count',tables['all']['counts']['roots'],'fallback',tables['all']['return_residual'],flush=True)


if __name__=='__main__':
    main()

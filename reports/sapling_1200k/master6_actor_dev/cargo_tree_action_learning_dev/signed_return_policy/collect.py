"""Reprice observed full returns on archived actions and construct strictly public model inputs."""
from collections import Counter,defaultdict
import hashlib,json,gzip,sys
from pathlib import Path
import numpy as np
from policy import context,NAMES

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
BASE=ACTOR.parent
ROOT=BASE.parents[1]
SIGNED=ACTOR/'joint_response_critic_dev'
TRANSFER='543a9d927ec9a2f6b6e2020ac2236eb9aecb594c0ee16a44c1a0ddee86f23797'
ORIGINAL='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
sys.path.insert(0,str(BASE))
from turning_point_ledger import CashEvents,FastCarbon,compact,write
from tree_win_gate import validate_examiner_configuration
sys.path.insert(0,str(BASE/'root_recovery_branches'))
from branches import begin, public

def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(compact(v)).hexdigest()
def clean(v):return {k:a for k,a in v.items() if a and a!='NONE'}
def rel(p):return str(p.relative_to(ROOT))
def load_tape(info,arm):
    ap=ROOT/info['action_file']
    assert sha(ap)==info['action_sha256'] and sha(ROOT/info['receipt'])==info['receipt_sha256']
    d=read(ap)
    if isinstance(d,list):return d
    if 'joint0' in d:return [d['joint0']]+[h['joint'] for h in d['history']]
    if 'orders' in d:return d['orders']
    part=d['arms']['v1' if arm=='parent' else 'candidate']
    return [part['first_joint'],part['last_joint']]

def catalog(signed):
    old={r['case']:r['identity'] for r in signed['rows']+signed['repeated_streams']}
    indexed=defaultdict(list)
    for row in signed['rows']:
        indexed[row['source_sha256'],row['seat'],row['t']].append(row)
    result=[];extras=[]
    for p in sorted(BASE.rglob('live_protocol.json')):
        folder=p.parent;case=rel(folder);prot=read(p);pa=read(folder/'live_parent.json')
        altfiles=[f for f in sorted(folder.glob('live_*.json')) if f.name not in ('live_parent.json','live_protocol.json','live_shared_init.json') and 'arm' in read(f)]
        assert len(altfiles)==1
        af,=altfiles;aa=read(af);pack=pa['pack_sha256'];seat=prot['seat'];t=prot['root_t']
        assert pa['pack_sha256']==aa['pack_sha256'] and pa['source_sha256']==aa['source_sha256']==ORIGINAL
        follow=[(f,read(f)) for f in sorted([*folder.glob('follow*.json'),*folder.glob('second_turn.json')])]
        override=any(clean(v['executed'])!=clean(v['parent']) for _,v in follow)
        log=dict(case=case,seed=prot['seed'],seat=seat,t=t,family=prot['family'],continuation_pack=pack,
                 signed_v1_root_id=old.get(case),source=prot.get('record',{}).get('source'),
                 followup_checks=[rel(f) for f,_ in follow],actual_secondary_override=override)
        result.append(log)
        if case in old or pack not in (TRANSFER,signed['frozen_parent_sha256']) or override:continue
        raw=read(ROOT/log['source']);arms={};actions={};scores={}
        for arm,f,a in [('parent',folder/'live_parent.json',pa),('alternative',af,aa)]:
            ap=folder/a['action_tape'];tape=read(ap)
            assert a['source_sha256']==ORIGINAL and a['protocol_sha256']==sha(p)
            assert a['shared_init_sha256']==sha(folder/'live_shared_init.json') and a['statuses']==['DONE','DONE']
            assert a['original_trajectory_sha256']==sha(ROOT/log['source'])==prot['record']['source_sha256']
            assert a['action_sha256']==sha(ap) and len(tape)==a['search_calls']+1 and tape[0]==a['root_actions']
            assert a['terminal_t']==t+len(tape)
            assert clean(tape[0][1-seat])==clean(raw['actions'][t][1-seat])
            arms[arm]=dict(action_file=rel(ap),action_sha256=sha(ap),receipt=rel(f),receipt_sha256=sha(f),
                           terminal_step=a['terminal_t'],calls=a['search_calls'])
            scores[arm+'_scores']=a['score'];actions[arm]=clean(tape[0][seat])
        assert pa['root_opponent_observation_sha256']==aa['root_opponent_observation_sha256']
        if actions['parent']!=clean(raw['actions'][t][seat]):
            log['excluded_from_transfer']='different source parent action';continue
        if actions['parent']==actions['alternative']:
            log['excluded_from_transfer']='no physical action change';continue
        changed=sorted(k for k in set(actions['parent'])|set(actions['alternative']) if actions['parent'].get(k)!=actions['alternative'].get(k))
        identical=[q for q in indexed[sha(ROOT/log['source']),seat,t] if q['parent']==actions['parent'] and q['alternative']==actions['alternative'] and q['root_opponent']==clean(raw['actions'][t][1-seat])]
        assert len(identical)<=1
        group='transfer' if pack==TRANSFER else ('later_repeat' if identical else 'later_option')
        ident=identical[0]['identity'] if identical else digest([sha(ROOT/log['source']),t,seat,actions['parent'],actions['alternative']])
        log.update(supplement_group=group,exact_old_intervention=identical[0]['identity'] if identical else None)
        extras.append(dict(case=case,seed=raw['seed'],seat=seat,t=t,family=prot['family'],source=log['source'],
                      source_sha256=sha(ROOT/log['source']),terrain=raw['initial_terrain_u8_sha256'],parent_sha256=pack,
                      root_opponent=clean(raw['actions'][t][1-seat]),changed_uids=changed,
                      arms=arms,role=None,split=group,identity=ident,**actions,**scores))
    return result,extras

def settle(sim,state,case,arm):
    seat=case['seat'];t=case['t'];info=case['arms'][arm];tape=load_tape(info,arm)
    assert len(tape)==info['calls']+1 and t+len(tape)==info['terminal_step']
    assert clean(tape[0][seat])==case[arm] and clean(tape[0][1-seat])==case['root_opponent']
    sim.set_state(state);sim.cash=CashEvents(sim.cash);flows=np.zeros((2,2))
    for i,j in enumerate(tape):
        sim.cash.events.clear();sim.step(*j)
        for kind,s,value,*_ in sim.cash.events:
            if kind=='own_delivery':
                flows[1,s]+=value
                if i<24:flows[0,s]+=value
    scores=case[arm+'_scores']
    assert sim.step_count==info['terminal_step']
    for s,x in enumerate(scores):
        assert abs(sim.cash[s]-x)<=.011 or (x==0 and not sim.player_workers[s] and sim.cash[s]<30), (case['case'],arm,s,x,sim.cash)
    doc=dict(deposits_24=flows[0].round(6).tolist(),deposits_terminal=flows[1].round(6).tolist(),official_scores=scores,original_calls_after_root=info['calls'])
    m=np.r_[flows[:,seat]-flows[:,1-seat],scores[seat]-scores[1-seat]]
    return m,doc

def main():
    registry=read(SIGNED/'signed_roots.json.gz')
    menu_doc=read(SIGNED/'native_inputs.json.gz')
    assert menu_doc['source_sha256']==sha(SIGNED/'signed_roots.json.gz') and not menu_doc['excluded']
    menus={r['case']:r for r in menu_doc['rows']}
    prefix=read(HERE.parent/'paired_response_prefix/signed.json.gz')
    assert all(r['free_from_preexisting_divergence_and_same_root_enemy'] for r in prefix['rows'])
    official={r['case']:r for r in prefix['rows']}
    census,extras=catalog(registry)
    cases=[{**r,'split':menus[r['case']]['split']} for r in registry['rows']]
    cases += [{**r,'split':'repeat'} for r in registry['repeated_streams']]+extras
    out=[]; features=[];groups=defaultdict(list)
    for r in cases:groups[r['source']].append(r)
    for source,rows in sorted(groups.items()):
        path=ROOT/source;raw=read(path);source_hash=sha(path);s=raw['candidate_seat']
        assert raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and not raw['errors']
        validate_examiner_configuration(raw['configuration'])
        sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        assert hashlib.sha256(bytes(map(int,sim.carbon))).hexdigest()==raw['initial_terrain_u8_sha256']
        arm_sim=FastCarbon(raw['game_configuration'])
        referee=begin(raw)
        for r in sorted(rows,key=lambda v:v['t']):
            t=r['t'];case=r['case']
            assert r['terrain']==raw['initial_terrain_u8_sha256'] and r['source_sha256']==source_hash and r['seat']==s and r['seed']==raw['seed']
            assert (raw['pack_sha256']==r['parent_sha256'] or r['split']=='transfer') and clean(raw['actions'][t][s])==r['parent']
            while sim.step_count<t:
                joint=raw['actions'][sim.step_count]
                sim.step(*joint);referee.step(joint)
            obs=public(referee,s);obs.pop('remainingOverageTime')
            assert digest(obs)==r.get('public_root_sha256',digest(obs)), (case,digest(obs),r.get('public_root_sha256'))
            roles={('base' if u in obs['players'][s][1] else obs['players'][s][2][u][2].lower()) for u in r['changed_uids']}
            role=next(iter(roles)) if len(roles)==1 else 'mixed'
            assert r['role'] in (None,role)
            arow=menus.get(case);available=(arow['menu_commands'][:] if arow else [r['parent'],r['alternative']])
            if arow:
                assert arow['signed_parent']==r['parent'] and arow['signed_alternative']==r['alternative']
                assert available[0]==r['parent']
            hi=len(available)
            if r['alternative'] not in available:available.append(r['alternative'])
            idx=available.index(r['alternative'])
            featurize=context(obs); start=len(features);features.extend(featurize(a) for a in available)
            state=sim.get_state()
            pm,p=settle(arm_sim,state,r,'parent');am,a=settle(arm_sim,state,r,'alternative')
            y=(am-pm).round(6).tolist()
            if arow:
                assert abs(official[case]['delta_official_margin']-y[2])<1.e-6
            out.append(dict(case=case,identity=r.get('identity',digest([source_hash,t,s,r['parent'],r['alternative']])),
                            split=r['split'],seed=r['seed'],terrain=r['terrain'],seat=s,t=t,role=role,family=r['family'],
                            source=source,source_sha256=source_hash,continuation_pack=r['parent_sha256'],
                            observed_parent_pack=raw['pack_sha256'],public_sha256=digest(obs),changed_uids=r['changed_uids'],
                            both_commands=[r['parent'],r['alternative']],both_receipts=r['arms'],
                            menu_start=start,free_menu_length=hi if arow else None,alternative=start+idx,target=y,
                            settlements=[p,a]))
    unique_transfer=[r['identity'] for r in out if r['split']=='transfer']
    assert len(unique_transfer)==len(set(unique_transfer))
    np.savez_compressed(HERE/'public_features.npz',features=np.array(features))
    write(HERE/'rows.json.gz',out,True);write(HERE/'inventory.json.gz',census,True)
    write(HERE/'data.json',dict(features=NAMES,feature_file=sha(HERE/'public_features.npz'),rows=sha(HERE/'rows.json.gz'),
          source_registry=sha(SIGNED/'signed_roots.json.gz'),source_menus=sha(SIGNED/'native_inputs.json.gz'),
          verified_prefixes=sha(HERE.parent/'paired_response_prefix/signed.json.gz'),experiment=sha(HERE/'protocol.json'),
          inventory=sha(HERE/'inventory.json.gz'),counts_by_split=dict(Counter(r['split'] for r in out)),
          by_continuation_pack=dict(Counter(r['continuation_pack'] for r in census))))
    print(len(out),len(features),Counter(r['split'] for r in out),flush=True)
if __name__=='__main__':main()

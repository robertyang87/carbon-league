"""Measure exact support for frozen full-backbone update scalars on original-opponent returns."""
from collections import Counter,defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import gzip,hashlib,importlib.util,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
CORPUS=HERE.parent
EXP=CORPUS.parent
ACTOR=EXP.parent
ROOT=ACTOR.parents[2]
sys.path.insert(0,str(ACTOR/'blended_actor_dev/serving'))
import build as repack
sys.path.insert(0,str(ACTOR.parent/'root_recovery_branches'))
from branches import begin,public
from tree_win_gate import validate_examiner_configuration

def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compact(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(compact(v)).hexdigest()
def write(p,v):
    raw=compact(v)
    p.write_bytes(gzip.compress(raw,mtime=0) if p.suffix=='.gz' else (json.dumps(v,indent=2,sort_keys=True)+'\n').encode())
def clean(d):return {k:v for k,v in d.items() if v and v!='NONE'}
def pathrel(p):return str(p.relative_to(ROOT))
def module(p,name):
    spec=importlib.util.spec_from_file_location('scalar_'+name,p)
    model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model);return model

def packages(protocol,directory):
    center_path=EXP/'candidate.py';center_weights=EXP/'candidate.npz'
    endpoint=CORPUS/'full.py';end_weights=CORPUS/'full.npz';fit=read(CORPUS/'fit.json')['arms']['full']
    assert sha(center_path)==protocol['freeze_parent'] and sha(endpoint)==fit['package_sha256']
    assert sha(endpoint).startswith(protocol['full_endpoint']) and sha(end_weights)==fit['weights_sha256']
    center=center_path.read_text()
    models={'center':module(center_path,'center')}
    meta={'center':{'sha256':sha(center_path)}}
    with np.load(center_weights) as c,np.load(end_weights) as end:
        assert set(c.files)==set(end.files)
        assert repack.substitute(center,repack.quantize(center_weights,c))==center
        assert repack.substitute(center,repack.quantize(end_weights,c))==endpoint.read_text()
        old={k:v.copy() for k,v in c.items() if k!='meta'}
        delta={k:end[k]-v for k,v in old.items()}
        for name,scale in protocol['scalars'].items():
            if not scale:continue
            weights={k:v+np.float32(scale)*delta[k] for k,v in old.items()}
            w=directory/(name+'.npz');np.savez(w,**weights,meta=c['meta'])
            p=directory/(name+'.py');p.write_text(repack.substitute(center,repack.quantize(w,c)))
            assert repack.substitute(p.read_text(),repack.literal(center,repack.STUDENT))==center
            model=module(p,name);models[name]=model
            meta[name]=dict(sha256=sha(p),bytes=p.stat().st_size,scalar=scale,
                            changed_fp32=sum(int(np.sum(weights[k]!=old[k])) for k in old))
    return models,dict(parent_weights_sha256=sha(center_weights),full_weights_sha256=sha(end_weights),
                       full_package_sha256=sha(endpoint),packages=meta)

def load_cases():
    prot=read(HERE/'protocol.json');parent=prot['freeze_parent'];cases=[];omitted=[]
    seen=set()
    def add(group,r,command,scores,suffix,registry,ident,**details):
        assert len(command)==len(scores)==2 and clean(command[0])!=clean(command[1])
        source=r['source'];path=ROOT/source;assert sha(path)==r['source_sha256']
        cases.append(dict(group=group,case=ident,source=source,source_sha256=r['source_sha256'],seed=r['seed'],
                          seat=r['seat'],t=r['t'],terrain=r['terrain'],archived_commands=list(map(clean,command)),
                          scores=scores,continuation=suffix,registry=pathrel(registry),**details))
    old=EXP/'signed_return_policy/rows.json.gz';results=read(EXP/'signed_return_policy/data.json');assert sha(old)==results['rows']
    for r in read(old):
        add('old_'+r['split'],r,r['both_commands'],[a['official_scores'] for a in r['settlements']],r['continuation_pack'],
            old,r['case'],public_expected=r['public_sha256'],repeat=r['split'] in ('repeat','later_repeat'))
        seen.add(r['case'])
    # Unextracted live sources are retained explicitly, including older continuations and extra overrides.
    for r in read(EXP/'signed_return_policy/inventory.json.gz'):
        if r['case'] not in seen:
            omitted.append(dict(group='older_live_registry',case=r['case'],reason='later extra active overrides' if r['actual_secondary_override'] else 'other historical continuation outside the reconciled 147 comparisons',continuation=r['continuation_pack']))
    b=EXP/'learned_carbon_observation'
    trained=b/'native_training';f=trained/'train_pairs.json';summary=read(f)
    assert summary['settlement']==sha(trained/'labels.json.gz')
    for r in summary['examples']:
        ident=r['root_id']
        for arm,key in [('parent','parent_receipt'),('student','proposal_receipt')]:
            assert sha(trained/('root_'+ident[:16])/(arm+'.json.gz'))==r[key]
        add('active_single',dict(r,t=r['step']),[r['parent'],r['proposal']],[r['parent_official_scores'],r['proposal_official_scores']],parent,f,ident,
            expected_prefix=r['prefix_actions_sha256'],public_expected=r['signed_observation_sha256'])
    dc=b/'decoder/real_roots';f=dc/'roots.json';vr=read(dc/'verification.json');assert vr['roots']==sha(f)
    for r,v in zip(read(f)['cases'],vr['cases'],strict=True):
        assert r['id']==v['id'];p=dc/r['prefix_file'];assert sha(p)==r['prefix_file_sha256']
        prefix=read(p)['physical_fixed_history']
        assert all(sha(dc/'native'/r['id']/(n+'.json.gz'))==v['arms'][n]['stored_ticket_sha256'] for n in ['parent','student'])
        add('decoder_single',r,[r['parent'],r['student']],[v['arms'][n]['official_scores'] for n in ['parent','student']],parent,
            dc/'verification.json',r['id'],expected_prefix=digest(prefix),public_expected=r['public_obs_sha256'])
    stream=b/'native_response';f=stream/'result.json';settled=read(f)
    for v in settled['maps']:
        ident=f"{v['seed']}_{v['seat']}"
        if not v['changed']:
            omitted.append(dict(group='encoder_single',case=ident,reason='no eligible changed command or response pair'));continue
        p=stream/ident/'shared.json.gz';d=read(p);case=d['case'];a=d['root'];s=case['seat']
        assert d['identity']['parent']==parent
        for name in ('parent','student'):
            ticket=v['arms'][name];assert ticket['shared']==sha(p) and ticket['receipt']==sha(stream/ident/(name+'.json.gz'))
            assert clean(ticket['root_orders'][s])==clean(a[name])
        r=dict(case,source=case['archived_map_fixture'],source_sha256=case['fixture_sha256'],t=a['t'])
        add('encoder_single',r,[a['parent'],a['student']],[v['arms'][n]['score'] for n in ['parent','student']],parent,
            f,ident,custom_prefix=pathrel(p),expected_prefix=digest(d['prefix']),public_expected=a['own_sha'])
    for batch in ('windows','earlier'):
        f=EXP/'paired_response_prefix'/(batch+'.json.gz')
        for r in read(f)['rows']:
            if r['group'] not in ('onpolicy','source_distill'):
                omitted.append(dict(group=r['group'],case=r['case'],reason='multi-turn treatment or prior divergence',prefix_verdict=r['prefix_verdict']));continue
            assert r['free_from_preexisting_divergence_and_same_root_enemy'] and r['start']==r['first_actual_intervention']
            dr=EXP/'onpolicy_native/response' if r['group']=='onpolicy' else ACTOR/'master3s2_policy_distill_dev/native_response'
            roots=read(dr/'roots.json');q=next(x for x in roots['cases'] if x['id']==r['case'])
            assert q['source_sha256']==r['source_sha256'] and q['root']['t']==r['start']
            add(r['group'],dict(r,t=r['start']),[q['root']['proposals'][n] for n in ['parent','student']],r['official_scores_physical'],roots['packs']['parent'],f,r['case'])
    for v in read(CORPUS/'original_response/settlement.json')['cases']:
        omitted.append(dict(group='trained_backbone_window',case=v.get('id',str(v.get('seed'))),reason='32-turn intervention changes continuation after first divergence'))
    omitted.append(dict(group='new_game_matrix',case='621758400,621758401 both seats',reason='eight whole games under different persistent policies and independently responding originals; no one-turn label'))
    return cases,omitted

def run(cases,models):
    splits=read(CORPUS/'sources.json')['split_by_terrain'];groups=defaultdict(list);out=[]
    for r in cases:groups[r['source'],r['seat'],r.get('custom_prefix')].append(r)
    for (source,s,prefix_override),rs in sorted(groups.items(),key=lambda v:str(v[0])):
        raw=read(ROOT/source);validate_examiner_configuration(raw['configuration'])
        assert raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1'
        env=begin(raw);history=raw['actions'] if not prefix_override else read(ROOT/prefix_override)['prefix']
        config={k:v for k,v in raw['game_configuration'].items() if k!='randomSeed'};at=defaultdict(list)
        for r in rs:
            assert r['terrain']==raw['initial_terrain_u8_sha256'] and r['seed']==raw['seed'];at[r['t']].append(r)
        for t in range(max(at)+1):
            obs=public(env,s)
            if t==0 or t in at:
                actions={name:clean(m.agent(obs,config)) for name,m in models.items()}
            else:
                for m in models.values():m._FEAT.observe(obs)
            for r in at[t]:
                full=digest(obs);short=digest({k:v for k,v in obs.items() if k!='remainingOverageTime'})
                if r.get('expected_prefix'):assert digest(history[:t])==r['expected_prefix'],(r['case'],'history')
                if r.get('public_expected'):assert r['public_expected'] in (full,short),(r['case'],full,short,r['public_expected'])
                frozen=r['archived_commands'];sco=r['scores'];signed=[round(a[s]-a[1-s],2) for a in sco]
                wdl=[float(np.sign(v)*.5+.5) for v in signed]
                index=lambda a:frozen.index(a) if a in frozen else None
                original=index(actions['center'])
                summaries={}
                for name,a in actions.items():
                    selected=index(a);changed=a!=actions['center'];both=selected is not None and original is not None
                    summaries[name]=dict(changed=changed,center_archived_arm=original,choice_archived_arm=selected,
                        category=('same_as_center' if not changed else ('both_observed' if both else 'changed_unmeasured')),
                        paired_wdl=(wdl[selected]-wdl[original]) if changed and both else (0. if not changed else None),
                        paired_official_margin=round(signed[selected]-signed[original],2) if changed and both else (0. if not changed else None))
                out.append(dict(**r,native_model_terrain_split=splits.get(r['terrain'],'outside_registry'),prefix_sha256=digest(history[:t]),
                                public_sha256=short,free_commands=actions,by_scalar=summaries))
            if t<max(at):env.step(history[t])
    return out

def main():
    prot=read(HERE/'protocol.json');cases,omitted=load_cases();write(HERE/'exclusions.json.gz',omitted)
    with TemporaryDirectory(prefix='sapling-direction-') as path:
        models,meta=packages(prot,Path(path));rows=run(cases,models)
        base=models['center']._NETS[1].w
        for name,m in models.items():
            meta['packages'][name]['changed_deployed_values']=sum(int(np.count_nonzero(base[k]!=m._NETS[1].w[k])) for k,shape,typ,nbytes in m._QMETA)
    write(HERE/'packages.json',meta);write(HERE/'commands.json.gz',rows)
    print('records',len(rows),len(omitted),flush=True)
if __name__=='__main__':main()

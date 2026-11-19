"""Dormant collector for a separately approved historical training root. A process never searches for replacement roots."""
import argparse
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;LEARN=HERE.parent
spec=importlib.util.spec_from_file_location('qualified_native_response_kernel',LEARN/'native_response/run.py')
K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
R=K.R;np=K.np

def replay_prefix(raw,card,s,t,arm_models,commands,public_hashes):
    env=K.checked_env(raw);cfg=raw['game_configuration'];trace=[]
    for i in range(t+1):
        a=K.public(env,s);op=K.public(env,1-s)
        if i==t:return env,a,op,trace
        want=commands[i];assert [K.digest(K.compact(a)),K.digest(K.compact(op))]==public_hashes[i]
        changes={}
        for name,mdl in arm_models.items():
            got,w=K.own_call(mdl,a,cfg);expected=want[1 if name=='parent' else 2]
            assert got==expected
            changes[name]=got
        pair=raw['actions'][i]
        trace.append(dict(t=i,own_sha=public_hashes[i][0],enemy_sha=K.digest(K.compact(op)),historical_joint=pair,proposed=changes))
        env.step(pair)

def run(idx,cpu):
    os.sched_setaffinity(0,{cpu}); pilot=R.load(HERE/'batch.json');summary=R.load(HERE/'summary.json')
    permit=R.load(HERE/'collection_authorized.json')
    assert permit['approved'] and permit['batch']==R.sha(HERE/'batch.json') and permit['plan']==R.sha(HERE/'collect_plan.json')
    assert permit['main_commit'] and permit['summary']==R.sha(HERE/'summary.json')
    assert permit['collector']==R.sha(Path(__file__)) and pilot['plan']==permit['plan']
    assert 0<len(pilot['roots'])<=4 and len({r['card']['terrain'] for r in pilot['roots']})==len(pilot['roots'])
    assert permit['root_ids']==[r['root_id'] for r in pilot['roots']]
    c=pilot['roots'][idx];mode=c['mode']
    assert mode in ('natural_argmax','active_supported_proposal')
    assert c['card'] in R.load(HERE/'protocol.json')['training_cards']
    src=R.ROOT/c['selection_path'];assert src.is_file() and R.sha(src)==c['selection_sha256']
    assert c['parent_sha']==K.digest(K.compact(c['parent'])) and c['student_sha']==K.digest(K.compact(c['student']))
    assert R.sha(R.ROOT/c['verification_path'])==c['verification_sha256']
    assert c['parent']!=c['student'] and c['no_previous_identical_root_label']
    card=c['card'];s=c['seat'];enemy=1-s;t=c['t'];case_id=c['root_id'];raw=R.load(R.ROOT/card['path'])
    assert R.sha(R.ROOT/card['path'])==card['sha256'] and raw['initial_terrain_u8_sha256']==card['terrain']
    assert R.sha(K.PARENT)==R.load(HERE/'protocol.json')['parent']
    assert R.sha(K.STUDENT)==K.STUDENT_SHA and R.sha(K.ORIGINAL)==R.load(LEARN/'native_response/protocol.json')['source_original_sha256']
    if mode=='natural_argmax':
        scope=R.load(HERE/'duplicates.json');flag=next(r for r in scope['flags'] if r['root_id']==case_id)
        assert flag['same_both_joint_orders']==0
    else:
        assert c['supported_contrast']>=1 and c['policy_probability']>=.08 and c['policy_rank']<=3 and c['joint_legal_verified']
    seatrecord=R.load(HERE/(str(card['seed'])+'.json.gz'))['seats'][s]
    assert seatrecord['seat']==s
    req=seatrecord['all_public_commands'];folder=HERE/('root_'+case_id[:16]);folder.mkdir(exist_ok=True)
    lock=(folder/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m,host=K.load_master(cpu);cfg=raw['game_configuration'];binding=dict(batch=permit['batch'],mode=mode,registration=R.sha(HERE/'protocol.json'),authorization=R.sha(HERE/'collection_authorized.json'),
        collector=R.sha(Path(__file__)),kernel=R.sha(Path(K.__file__)),parent=R.sha(K.PARENT),student=R.sha(K.STUDENT),examiner=R.sha(K.ORIGINAL),source=card['sha256'],root=case_id,cpu=cpu)
    root_file=folder/'root.json.gz'
    if root_file.exists():
        shared=R.load(root_file);assert shared['binding']==binding
    else:
        started=folder/'started.json';assert not started.exists(),'An existing unanswered natural root may not be resampled.'
        R.write(started,dict(binding=binding,utc=time.time()))
        env=K.checked_env(raw);tracker=m.BeliefTracker(m._Bel,enemy);prev=None;refs=[]
        own_models={k:R.live.module(path,'training_prefix_'+k) for k,path in [('parent',K.PARENT),('student',K.STUDENT)]}
        F=sys.modules['cnet2_feat'];legal_feat=F.Feat(s)
        for i in range(t):
            my=K.public(env,s);opp=K.public(env,enemy);actual=raw['actions'][i]
            assert [K.digest(K.compact(my)),K.digest(K.compact(opp))]==c['full_public_hashes'][i]
            legal_feat.observe(my)
            for n,mdl in own_models.items():assert K.own_call(mdl,my,cfg)[0]==req[i][1 if n=='parent' else 2]
            tracker.step(opp['players'],i,opp['carbon']);tracker.set_actions(opp['players'],actual[enemy]);prev=opp['players']
            refs.append(c['full_public_hashes'][i]);env.step(actual)
        my=K.public(env,s);opp=K.public(env,enemy);assert my['step']==t and K.digest(K.compact({k:v for k,v in my.items() if k!='remainingOverageTime'}))==c['public_sha256']
        if t:
            m._ST.clear();m._ST.update(tr=tracker,prev=prev,rng=np.random.RandomState(None))
        rec=R.native_clock.Recorder(m);native=rec.wrap(m.agent)
        original=R.live.clean(native(opp,cfg));frozen=K.capture(m);sig=K.sig(m)
        ownacts={k:K.own_call(mdl,my,cfg)[0] for k,mdl in own_models.items()}
        assert ownacts['parent']==c['parent'] and ownacts['student']==req[t][2]==c['expected_free_student']
        if mode=='natural_argmax':assert ownacts['student']==c['student']
        actual_own_normal=ownacts.copy()
        ownacts['student']=c['student']
        # Even for an explicitly supported (non-argmax) proposal, verify the entire actual physical joint is legal
        # at its own sequential prefix and serializes back to precisely the registered commands.
        f=legal_feat;x=f.observe(my)
        for proposal in ownacts.values():
            labels=f.labels(proposal);pre=F.Prefix(x,f.aux)
            for sl in np.flatnonzero(x['tok_mask'][:F.OWN]):
                _,_,legal=pre.feats(sl);assert labels[sl]>=0 and legal[labels[sl]]
                pre.advance(sl,int(labels[sl]))
            assert R.live.clean(f.to_actions(labels))==proposal
        K.restore(m,frozen);assert K.sig(m)==sig
        K.blob_write(folder/'root_state.pickle.gz',gzip.compress(frozen,mtime=0))
        shared=dict(binding=binding,seed=card['seed'],seat=s,t=t,terrain=card['terrain'],host=host,public_prefix_sha=K.digest(K.compact(refs)),
                    historical_prefix_joint_sha=K.digest(K.compact(raw['actions'][:t])),own_obs_sha=K.digest(K.compact(my)),enemy_obs_sha=K.digest(K.compact(opp)),
                    original=original,archived_enemy=c['historical_enemy'],own=ownacts,actual_normal=actual_own_normal,snapshot=K.digest(frozen),state_sig=sig,clock=K.clock_result(rec,m))
        R.write(root_file,shared)
    for arm in (('parent','student') if idx%2==0 else ('student','parent')):
        out=folder/(arm+'.json.gz')
        if out.exists():
            assert R.load(out)['binding']==binding
            continue
        tic=time.perf_counter();model=R.live.module(K.PARENT,'training_factual_history_'+arm)
        env,my,opp,refs=replay_prefix(raw,card,s,t,{'parent':model},req,c['full_public_hashes'])
        assert K.digest(K.compact([[d['own_sha'],d['enemy_sha']] for d in refs]))==shared['public_prefix_sha']
        assert K.own_call(model,my,cfg)[0]==c['parent'] and K.digest(K.compact(opp))==shared['enemy_obs_sha']
        frozen=gzip.decompress((folder/'root_state.pickle.gz').read_bytes());assert K.digest(frozen)==shared['snapshot'];K.restore(m,frozen);assert K.sig(m)==shared['state_sig']
        rec=R.native_clock.Recorder(m);native=rec.wrap(m.agent)
        tape=[K.physical(shared['own'][arm],shared['original'],s)];env.step(tape[0]);obs_hash=[]
        while not env.done:
            my=K.public(env,s);opp=K.public(env,enemy);obs_hash.append([K.digest(K.compact(my)),K.digest(K.compact(opp))])
            en=R.live.clean(native(opp,cfg));a,w=K.own_call(model,my,cfg);tape.append(K.physical(a,en,s));env.step(tape[-1])
        scores,cash,status=K.score(env);clock=K.clock_result(rec,m);endhost=R.native_clock.host_snapshot(m);R.native_clock.validate_host(endhost,[cpu])
        assert len(rec.calls)==len(tape)-1
        R.write(out,dict(binding=binding,arm=arm,root=c,root_receipt=R.sha(root_file),state_sig=shared['state_sig'],tape=tape,tape_sha=K.digest(K.compact(tape)),suffix_public_sha=obs_hash,
               scores=scores,cash=cash,margin=scores[s]-scores[enemy],status=status,clock=clock,wall_s=time.perf_counter()-tic,host_at_end=endhost))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--idx',type=int,required=True);p.add_argument('--cpu',type=int,required=True);a=p.parse_args();run(a.idx,a.cpu)

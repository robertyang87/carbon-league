"""Two matched eight-turn teacher interventions from the same sealed public prefix, with full challenger reponses."""
from pathlib import Path
import sys,json,gzip,time,os,hashlib,subprocess,fcntl
import numpy as np
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parents[1];BASE=ACTOR.parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/'root_recovery_branches'))
import live
from branches import clean,public,begin
def read(p):
    b=p.read_bytes();return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
P=read(HERE/'protocol.json');PACK=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py';TREE=ACTOR/'tree_response_training'
def digest(obs):return hashlib.sha256(json.dumps(obs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def install(path,side,seed,cost,name):
    m=live.module(path,name);assert m._SEED is None and m._TEMP==1 and m._RX and m._SEARCH_OK and m._S_ABORT==.58
    m._S_COST[:]=cost;m._ST.clear();m._ST.update(tr=m.BeliefTracker(m._Bel,side),prev=None,rng=np.random.RandomState(seed));return m
def historical(m,obs,cmd):
    assert set(obs)=={'player','step','players','trees','carbon','remainingOverageTime'}
    assert set(cmd)<=set(obs['players'][obs['player']][1])|set(obs['players'][obs['player']][2])
    m._ST['tr'].step(obs['players'],obs['step'],obs['carbon']);m._ST['tr'].set_actions(obs['players'],cmd);m._ST['prev']=obs['players']
def call(m,obs,cfg):
    assert 'full_carbon' not in obs
    stats=m._S_STATS;names=('searched','single','changed','aborted')
    prev={k:stats[k] for k in names};last=stats.get('last');w=time.monotonic();cpu=time.process_time()
    a=clean(m.agent(obs,cfg));cw=time.monotonic()-w;cc=time.process_time()-cpu
    z=stats.get('last');new=z is not None and z is not last
    val=dict(step=obs['step'],wall=cw,cpu=cc,delta={k:stats[k]-prev[k] for k in names},last=(list(z[:3]) if new else None),zero_depth=(new and z[1]==0),public_digest=digest(obs))
    return a,val
def run(i,arm):
    c=P['cases'][i];s=c['seat'];op=1-s
    assert os.sched_getaffinity(0)=={c['cpu']} and os.environ.get('CARBON_DEBUG')=='1'
    start=time.monotonic();src=ROOT/c['source'];warm=ROOT/c['warm_source']
    assert sha(src)==c['source_sha'] and sha(warm)==c['warm_sha'] and sha(PACK)==P['identity']['parent'] and sha(live.SOURCE)==P['identity']['master']==hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:sources/master3s2/master3s2.py'])).hexdigest()
    costs=read(warm)['warm_cost'];raw=read(src);cfg=raw['game_configuration'];assert raw['pack_sha256']==sha(PACK) and raw['candidate_seat']==s and raw['initial_terrain_u8_sha256']==c['terrain']
    env=begin(raw);own=live.module(PACK,'focal_it50');enemy=install(live.SOURCE,op,c['enemy_stream_seed'],costs,'enemy_original')
    teacher=install(live.SOURCE,s,c['teacher_stream_seed'],costs,'teacher_original') if arm=='teacher' else None
    for t,pair in enumerate(raw['actions'][:150]):
        so=public(env,s);eo=public(env,op)
        historical(enemy,eo,pair[op])
        if teacher:historical(teacher,so,pair[s])
        if t==0:assert clean(own.agent(so,cfg))==clean(pair[s])
        else:own._FEAT.observe(so)
        env.step(pair)
    checks=[];opclock=[];teclock=[];tape=[];snap=[];f=HERE/f'{c["seed"]}_{s}_{arm}'
    while not env.done:
        t=env.states[0].observation.step;so=public(env,s);eo=public(env,op)
        if t%25==0 or t in (158,168):snap.append(dict(t=t,own_enemy_cash=[so['players'][j][0] for j in (s,op)],own_enemy_trees=[len(so['players'][j][3]) for j in (s,op)]))
        base=clean(own.agent(so,cfg));a=base
        if teacher and t<=157:
            a,log=call(teacher,so,cfg);teclock.append(log)
        if t<=157:checks.append(dict(t=t,baseline=base,submitted=a,own_observation_sha=digest(so)))
        if t==150:
            assert base==clean(raw['actions'][t][s])
            b=clean(raw['actions'][t][op]);historical(enemy,eo,b);root_ob=digest(eo)
        else:
            b,log=call(enemy,eo,cfg);opclock.append(log)
        acts=[a,b] if s==0 else [b,a]
        env.step(acts);tape.append(acts)
        if time.monotonic()-start>220:
            live.save(Path(str(f)+'_incomplete.json.gz'),dict(tape=tape,checks=checks,enemy=opclock,teacher=teclock));raise TimeoutError('predeclared wall clock')
    scores=[x.reward for x in env.states];assert [x.status for x in env.states]==['DONE','DONE'] and len(checks)==8 and len(opclock)==len(tape)-1
    if teacher:assert len(teclock)==8
    out=Path(str(f)+'.json.gz')
    live.save(out,dict(seed=c['seed'],seat=s,arm=arm,scores=scores,margin=scores[s]-scores[op],tape=tape,checks=checks,enemy=opclock,teacher=teclock,
                        wall=time.monotonic()-start,cpu_affinity=list(os.sched_getaffinity(0)),host_load=os.getloadavg(),root_op_sha=root_ob,
                        protocol=sha(HERE/'protocol.json'),program=sha(Path(__file__)),snap=snap))
    print(c['seed'],s,arm,scores,flush=True)
def dispatch(i):
    for a in (('parent','teacher') if i==0 else ('teacher','parent')):
        subprocess.run([sys.executable,__file__,str(i),a],check=True,timeout=230)
if __name__=='__main__':
    if len(sys.argv)==3:run(int(sys.argv[1]),sys.argv[2])
    elif len(sys.argv)==2:dispatch(int(sys.argv[1]))
    else:
        with (TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not list(HERE.glob('621*json.gz')),'prior bundle exists; inspect; never rerun blind'
            jobs=[subprocess.Popen(['taskset','-c',str(i),sys.executable,__file__,str(i)]) for i in (0,1)]
            assert [p.wait(timeout=500) for p in jobs]==[0,0]

from pathlib import Path
import sys,os,json,subprocess,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run as native
B=native;P=B.read(HERE/'protocol.json');ROOT=B.ROOT
sys.path.insert(0,str(B.ACTOR/'master3s2_policy_distill_dev'))
import builder as panel
def collect(c):
    path=HERE/f"card_{c['id']:02d}.json.gz"
    if path.exists():
        prior=B.read(path);assert prior['protocol']==B.sha(HERE/'protocol.json') and len(prior['records'])==8;return
    src=ROOT/c['source'];warm=ROOT/c['warm'];assert B.sha(src)==c['sha'] and B.sha(warm)==c['warm_sha']
    raw=B.read(src);s=c['seat'];cfg=raw['game_configuration'];sim=panel.FastCarbon(cfg).reset(seed=c['seed'])
    assert raw['pack_sha256']==B.P['identity']['parent'] and B.sha(B.live.SOURCE)==B.P['identity']['master']
    seed=int(np.random.SeedSequence([4321901,c['seed'],s]).generate_state(1)[0]);reuse=None
    if c['reuse_sha']:
        old=HERE.parent/f"{c['seed']}_{s}_teacher.json.gz";assert B.sha(old)==c['reuse_sha'];reuse=B.read(old)
        seed=next(e['teacher_stream_seed'] for e in B.P['cases'] if e['seed']==c['seed'] and e['seat']==s)
    m=B.install(B.live.SOURCE,s,seed,B.read(warm)['warm_cost'],'original_demonstrator');rec=[]
    for t,a in enumerate(raw['actions'][:158]):
        obs=sim.obs(s);obs={k:obs[k] for k in ('player','step','players','trees','carbon','remainingOverageTime') if k in obs};obs.setdefault('remainingOverageTime',60.)
        if t<150:B.historical(m,obs,a[s])
        else:
            if t==150 and reuse:
                cmd=reuse['checks'][0]['submitted'];func=m._search;m._search=lambda *unused:cmd
                assert B.clean(m.agent(obs,cfg))==cmd;m._search=func;clock=dict(reused=True)
            else:cmd,clock=B.call(m,obs,cfg)
            m._ST['tr'].set_actions(obs['players'],a[s]);rec.append(dict(t=t,executed=a[s],teacher=cmd,clock=clock,public_hash=B.digest({k:v for k,v in obs.items() if k!='remainingOverageTime'})))
        sim.step(*a)
    B.live.save(path,dict(records=rec,protocol=B.sha(HERE/'protocol.json'),id=c['id'],source=c['sha'],rng=seed))
if __name__=='__main__':
    side=int(sys.argv[1]);assert os.sched_getaffinity(0)=={2+side};os.environ['CARBON_DEBUG']='1'
    for card in P['cards']:
        if card['seat']==side:collect(card)

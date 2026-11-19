"""Establish exact parent action equality on the exposed prefix, then run that parent on both branches."""
from response import HERE,sr,runner
import sys,subprocess,fcntl

def prepare():
    src=HERE/'seat1';out=HERE/'exact_parent/seat1';out.mkdir(parents=True,exist_ok=True)
    q=sr.load(src/'live_protocol.json');raw=sr.load(sr.ROOT/q['record']['source'])
    pack=HERE.parent/'planter_capture_full_dev/candidate.py'; actor=sr.live.module(pack,'toll_prefix_parent')
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
    checked=0
    for t,pair in enumerate(raw['actions'][:q['root_t']+1]):
        orders=actor.agent(sim.obs(1),sim.config)
        assert orders==pair[1],f'source and intended parent differ at {t}'
        sim.step(*pair);checked+=1
    sr.retain(out/'prefix.json',dict(steps_checked=checked,logged_pack=raw['pack_sha256'],verified_parent=sr.sha(pack)))
    sr.retain(out/'official_replay.json',sr.load(src/'official_replay.json'))
    sr.retain(out/'live_shared_init.json',sr.load(src/'live_shared_init.json'))
    sr.retain(out/'live_protocol.json',{**q,'family':'verified_equal_parent_prefix','source_dir':str(out.relative_to(sr.ROOT)),'pack_sha256':sr.sha(pack)})
def play(arm):
    runner.HERE=HERE/'exact_parent';runner.PRIOR=HERE.parent/'planter_capture_full_dev';runner.play(1,arm)
if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:play(sys.argv[1])
    else:
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            jobs=[subprocess.Popen(['taskset','-c',str((i+1)%2),sys.executable,__file__,arm]) for i,arm in enumerate(('parent','approach'))]
            if any([j.wait() for j in jobs]):raise RuntimeError('original parent replication failed')

import sys,subprocess,fcntl,inspect
from role import HERE,choose
sys.path.insert(0,str(HERE.parent/'capture_pressure_branches'))
import compare as runner
sr=runner.sr;live=sr.live;PACK=HERE.parent/'planter_capture_full_dev'
def prepare():
    selections=[]
    for seat,folder in enumerate(('capture_reach_dev/games/parent','intercept_response_dev/games/parent')):
        path,=list((HERE.parent/folder).glob(f'raw_*_{seat}.json.gz'));raw=sr.load(path);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        for t,pair in enumerate(raw['actions']):
            r=choose(sim.obs(seat),pair[seat],raw['game_configuration']) if 120<=t<=200 else None
            if r:break
            sim.step(*pair)
        else:raise ValueError('no eligible base')
        out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
        receipt=sr.load(path.with_name(path.name.replace('raw_','receipt_').replace('.json.gz','.json')))
        assert sr.sha(path)==receipt['game']['raw_sha256'] and sr.sha(PACK/'candidate.py')==raw['pack_sha256']
        sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[receipt['official']]}}})
        sr.retain(out/'live_protocol.json',dict(seed=raw['seed'],seat=seat,root_t=t,family='midgame_role',source_dir=str(out.relative_to(sr.ROOT)),
            record=dict(seed=raw['seed'],seat=seat,source=str(path.relative_to(sr.ROOT)),source_sha256=sr.sha(path)),root=r,pack_sha256=raw['pack_sha256']))
        live.prepare(out);selections.append(r)
    sr.retain(HERE/'selection.json',selections)
def play(seat,arm):
    # The shared audited worker driver accepts moves. Extend only its root type and accepted command for a base.
    src=inspect.getsource(live.run)
    for old,new in [('r["worker"] not in own_obs["players"][seat][2]','r["worker"] not in own_obs["players"][seat][1]'),
                    ('r["tree"] in after.players[s][3]','False'),
                    ('r["alternative"] in ("UP","RIGHT","DOWN","LEFT")','r["alternative"]=="RECCOLLECTOR"')]:
        assert src.count(old)==1;src=src.replace(old,new)
    exec(src,live.__dict__)
    runner.HERE=HERE;runner.PRIOR=PACK;runner.play(seat,arm)
if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:play(int(sys.argv[1]),sys.argv[2])
    else:
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for seat in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(('parent','approach'))]
                if any([p.wait() for p in jobs]):raise RuntimeError('failed arm')

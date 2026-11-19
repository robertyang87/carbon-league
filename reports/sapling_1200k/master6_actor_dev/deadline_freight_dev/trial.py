from pathlib import Path
import sys,fcntl,subprocess
from rescue import HERE,choose
sys.path.insert(0,str(HERE.parent/'capture_pressure_branches'))
import compare as runner
sr=runner.sr
PACK=HERE.parent/'planter_capture_full_dev'
def prepare():
    audit=sr.load(HERE.parent/'bank_under_pressure_dev/audit.json')
    for c in audit:
        if not c['losses']:continue
        end,uid,*_=c['losses'][0]; path=sr.ROOT/c['source'];raw=sr.load(path);seat=c['seat']
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
        for t,pair in enumerate(raw['actions']):
            root=choose(sim.obs(seat),pair[seat]) if end-10<=t<=end else None
            if root and root['worker']==uid:break
            sim.step(*pair)
        else:
            sr.retain(HERE/f'miss_seat{seat}.json',dict(seed=c['seed'],victim=uid,t=end,selection='no public trigger on this victim in the ten preceding turns or collision turn'))
            continue
        out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
        receipt=sr.load(path.with_name(path.name.replace('raw_','receipt_').replace('.json.gz','.json')))
        assert sr.sha(path)==receipt['game']['raw_sha256']==c['sha256']
        sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[receipt['official']]}}})
        sr.retain(out/'live_protocol.json',dict(seed=c['seed'],seat=seat,root_t=t,family='deadline_freight',source_dir=str(out.relative_to(sr.ROOT)),
            record=dict(seed=c['seed'],seat=seat,source=c['source'],source_sha256=c['sha256']),root=root,pack_sha256=sr.sha(PACK/'candidate.py')))
        sr.live.prepare(out)

def play(seat,arm):
    runner.HERE=HERE;runner.PRIOR=PACK
    q=sr.load(HERE/f'seat{seat}'/'live_protocol.json')
    original=sr.live.module
    def loader(path,name):
        mod=original(path,name)
        if name=='recovery_live_student' and arm=='approach':
            parent=mod.agent
            def act(obs,cfg):
                orders=parent(obs,cfg)
                if obs['step']>q['root_t']:
                    move=choose(obs,orders,q['root']['worker'])
                    if move:orders=dict(orders,**{move['worker']:move['alternative']})
                return orders
            mod.agent=act
        return mod
    sr.live.module=loader;runner.play(seat,arm)

if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:play(int(sys.argv[1]),sys.argv[2])
    else:
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for seat in (0,1):
                if not (HERE/f'seat{seat}/live_protocol.json').exists():continue
                jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(('parent','approach'))]
                if any([p.wait() for p in jobs]):raise RuntimeError('worker failure')

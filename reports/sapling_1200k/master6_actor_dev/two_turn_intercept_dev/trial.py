from pathlib import Path
import sys,fcntl,subprocess
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'capture_pressure_branches'))
import compare as runner
from route import choose,finish
sr=runner.sr;live=sr.live
PACK=HERE.parent/'planter_capture_full_dev'

def prepare():
    selections=[]
    for seat,folder in enumerate(('planter_capture_full_dev/games/candidate','capture_reach_dev/games/parent')):
        source=HERE.parent/folder
        path,=list(source.glob(f'raw_*_{seat}.json.gz'))
        raw=sr.load(path);assert raw['pack_sha256']==sr.sha(PACK/'candidate.py')
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        for t,pair in enumerate(raw['actions']):
            root=choose(sim.obs(seat),pair[seat]) if 230<=t<=280 else None
            if root:break
            sim.step(*pair)
        if not root:raise ValueError(f'no eligible root for seat {seat}')
        out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
        receipt=sr.load(source/path.name.replace('raw_','receipt_').replace('.json.gz','.json'))
        assert receipt['game']['raw_sha256']==sr.sha(path)
        sr.retain(out/'official_replay.json',dict(runs={out.name:dict(games=[receipt['official']])}))
        case=dict(seed=raw['seed'],seat=seat,root_t=t,family='two_turn_intercept',source_dir=str(out.relative_to(sr.ROOT)),
                  record=dict(seed=raw['seed'],seat=seat,source=str(path.relative_to(sr.ROOT)),source_sha256=sr.sha(path)),
                  root=dict(**root,tree=''),pack_sha256=raw['pack_sha256'],route_sha256=sr.sha(HERE/'route.py'))
        sr.retain(out/'live_protocol.json',case);live.prepare(out);selections.append(root)
    sr.retain(HERE/'selection.json',selections)

def play(seat,arm):
    q=sr.load(HERE/f'seat{seat}'/'live_protocol.json')
    assert sr.sha(HERE/'route.py')==q['route_sha256']
    runner.HERE=HERE;runner.PRIOR=PACK
    orig=live.module
    def module(p,name):
        mod=orig(p,name)
        if name=='recovery_live_student' and arm=='approach':
            agent=mod.agent
            def act(obs,cfg):
                result=agent(obs,cfg)
                if obs['step']==q['root_t']+1:
                    revised=finish(obs,result,q['root'])
                    sr.retain(HERE/f'seat{seat}'/'second_turn.json',dict(parent=result,executed=revised))
                    return revised
                return result
            mod.agent=act
        return mod
    live.module=module
    runner.play(seat,arm)

if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:play(int(sys.argv[1]),sys.argv[2])
    else:
        with (runner.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for seat in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(('parent','approach'))]
                codes=[j.wait() for j in jobs]
                if any(codes):raise RuntimeError(codes)

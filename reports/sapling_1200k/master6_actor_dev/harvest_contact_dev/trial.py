from pathlib import Path
import sys,fcntl,subprocess
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'capture_pressure_branches'))
import compare as runner
from contact import choose,_TO,_DIRECTION
from occupy import harvest
sr=runner.sr;live=sr.live
PACK=HERE.parent/'planter_capture_full_dev'

def prepare():
    selections=[]
    for seat,folder in enumerate(('bank_under_pressure_dev/games/parent','deadline_freight_dev/games/parent')):
        source=HERE.parent/folder
        path,=list(source.glob(f'raw_*_{seat}.json.gz'))
        raw=sr.load(path);assert raw['pack_sha256']==sr.sha(PACK/'candidate.py')
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        prev=None
        for t,pair in enumerate(raw['actions']):
            obs=sim.obs(seat);root=choose(obs,pair[seat],prev) if 120<=t<=200 else None;prev=obs
            if root:root['site']=obs['players'][1-seat][2][root['target']][0]
            if root:break
            sim.step(*pair)
        if not root:raise ValueError(f'no eligible root for seat {seat}')
        out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
        receipt=sr.load(source/path.name.replace('raw_','receipt_').replace('.json.gz','.json'))
        assert receipt['game']['raw_sha256']==sr.sha(path)
        sr.retain(out/'official_replay.json',dict(runs={out.name:dict(games=[receipt['official']])}))
        case=dict(seed=raw['seed'],seat=seat,root_t=t,family='harvest_contact',source_dir=str(out.relative_to(sr.ROOT)),
                  record=dict(seed=raw['seed'],seat=seat,source=str(path.relative_to(sr.ROOT)),source_sha256=sr.sha(path)),
                  root=root,pack_sha256=raw['pack_sha256'],route_sha256=sr.sha(HERE/'occupy.py'))
        sr.retain(out/'live_protocol.json',case);live.prepare(out);selections.append(root)
    sr.retain(HERE/'selection.json',selections)

def play(seat,arm):
    q=sr.load(HERE/f'seat{seat}'/'live_protocol.json')
    assert sr.sha(HERE/'occupy.py')==q['route_sha256']
    runner.HERE=HERE;runner.PRIOR=PACK
    orig=live.module
    def module(p,name):
        mod=orig(p,name)
        if name=='recovery_live_student' and arm=='approach':
            agent=mod.agent
            def act(obs,cfg):
                result=agent(obs,cfg)
                if obs['step']==q['root_t']+1:
                    revised=harvest(obs,result,q['root'])
                    sr.retain(HERE/f'seat{seat}'/f"follow_{obs['step']}.json",dict(parent=result,executed=revised))
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

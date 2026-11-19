"""One real first-action counterfactual on two terrains withheld from the score-function fit."""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from experiment import ACTOR, FIT, PARENT, ROOT, read, sha
spec=importlib.util.spec_from_file_location("existing_two_arm_original_response",
        ACTOR/"master3s2_policy_distill_dev/native_response/run.py")
base=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=base
spec.loader.exec_module(base)
base.HERE=HERE
base.P={"parent":PARENT,"student":HERE.parent/"candidate.py"}
base.idle=lambda m, obs: m._FEAT.observe(obs)

def select():
    fit=read(HERE.parent/'fit.json')
    protocol=read(HERE/'protocol.json')
    if (fit['gate']!='eligible_for_old_root_response' or
        sha(HERE/'response.py')!=protocol['program'] or sha(Path(base.__file__))!=protocol['source_runner'] or
        sha(HERE.parent/'protocol.json')!=protocol['training_protocol']):
        raise ValueError('unqualified student or changed procedure')
    source=read(HERE.parent/'protocol.json')['held_command_audit']
    cases=[]
    for seed,seat in protocol['map_seats']:
        c=next(v for v in source if v['seed']==seed and v['seat']==seat)
        raw=read(ROOT/c['raw'])
        if sha(ROOT/c['raw'])!=c['sha256']:raise ValueError('unregistered source')
        world=base.R.FastCarbon(raw['game_configuration']).reset(seed=seed)
        models={k:base.R.live.module(p,'signed_sapling_'+k) for k,p in base.P.items()}
        found=None
        for t,joint in enumerate(raw['actions'][:251]):
            obs=world.obs(seat)
            if t==0 or t>=135:
                cmds={k:base.R.live.clean(m.agent(obs,world.config)) for k,m in models.items()}
                if t>=135 and cmds['parent']!=cmds['student']:
                    found=dict(t=t,proposals=cmds);break
            else:
                for m in models.values():base.idle(m,obs)
            world.step(*joint)
        cases.append(dict(id=f'm{seed}_s{seat}',seed=seed,seat=seat,source=c['raw'],
            terrain=c['terrain'],source_sha256=c['sha256'],root=found))
    base.R.write(HERE/'roots.json',dict(protocol=sha(HERE/'protocol.json'),runner=sha(Path(base.__file__)),
        selector=sha(Path(__file__)),training_fit=sha(HERE.parent/'fit.json'),
        cases=cases,packs={k:sha(p) for k,p in base.P.items()}))
    print([(c['id'],c['root'] and c['root']['t']) for c in cases],flush=True)

def run():
    # The reused runner checks its own source identity, frozen roots, pack hashes and clock.
    # It puts each alternate physical command into the same root, then holds continuation fixed.
    r=read(HERE/'roots.json')
    if r['selector']!=sha(Path(__file__)) or r['training_fit']!=sha(HERE.parent/'fit.json'):
        raise ValueError('response frozen-pack binding changed')
    base.run()

if __name__=='__main__':
    {'select':select,'run':run}[sys.argv[1]]()

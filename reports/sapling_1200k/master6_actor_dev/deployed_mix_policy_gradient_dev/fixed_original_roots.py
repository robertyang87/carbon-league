"""Conditionally bind the two preregistered archived boards; actual original calls require their shared native slot."""
from concurrent.futures import ProcessPoolExecutor
import argparse,fcntl,importlib.util,sys
from study import HERE,ACTOR,PLAN,PARENT,arena
upstream=ACTOR/'economic_multitask_backbone_dev'
spec=importlib.util.spec_from_file_location('pg_native',upstream/'native_responses.py')
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
native.HERE=HERE
native.PACK={'bc':PARENT,'economy':HERE/'candidate.py'}
def prepare():
    dev=arena.read(HERE/'development.json')
    if not dev['eligible']:raise ValueError('full-map development gate failed')
    source=arena.read(upstream/'native_conditions.json')['maps']
    if [(c['seed'],c['seat']) for c in source]!=[(621758320,0),(621758335,1)]:raise ValueError('changed roots')
    cases=[native.choice(c) for c in source]
    arena.write(HERE/'original_root_bindings.json',dict(cases=cases,policy_protocol=arena.sha(HERE/'protocol.json'),
        accepted_update=arena.sha(HERE/'update.json'),full_development=arena.sha(HERE/'development.json'),
        candidate=arena.sha(HERE/'candidate.py'),parent=arena.sha(PARENT),subsequent_exact_source=native.r.sha(native.r.live.SOURCE),
        fixture='First real differing command in t100-199 on each archived seat; fixed shared historical past, then live full decision both sides'))
def play():
    frozen=arena.read(HERE/'original_root_bindings.json')
    if frozen['full_development']!=arena.sha(HERE/'development.json') or frozen['candidate']!=arena.sha(HERE/'candidate.py'):
        raise ValueError('changed frozen evidence')
    cases=frozen['cases'];results=[]
    if any(c['t'] is None for c in cases):raise ValueError('no paired two-seat mechanism')
    with native.r.LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for c in cases:
            with ProcessPoolExecutor(1) as pool:warm=pool.submit(native.r.native_warm).result()
            with ProcessPoolExecutor(2) as pool:
                old,new=pool.map(native.play,[(c,'bc',c['seat'],warm),(c,'economy',1-c['seat'],warm)])
            results.append(dict(seed=c['seed'],seat=c['seat'],t=c['t'],parent_margin=old['margin'],candidate_margin=new['margin'],
                delta_margin=new['margin']-old['margin'],delta_u=new['U']-old['U'],original_calls=[len(a['calls']) for a in (old,new)]))
    arena.write(HERE/'original_root_outcomes.json',results)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','play']);arg=p.parse_args()
    prepare() if arg.mode=='prepare' else play()

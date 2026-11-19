"""Single-CPU, paired terminal response and independent real-funds audit on the two frozen interventions."""
import gzip,importlib.util,json,os,sys,time
from pathlib import Path
from analyze import BASE,ROOT,HERE,read,digest,save,PLAN
from policy import choose_intervention
sys.path.insert(0,str(BASE/'root_recovery_branches'))
from branches import begin,public,clean
import live

def cfg(name):
    reg=read(HERE/'roots.json');case=reg['maps'][name];wd=HERE/name;wd.mkdir(exist_ok=True)
    if case['selected'] is None:return None
    return reg,case,case['selected'],case['map'],wd
def prepare(name):
    if not cfg(name):return
    reg,ci,r,role,wd=cfg(name);raw=read(ROOT/r['source']);s=r['seat']
    env=begin(raw);prefix=[];own=live.module(ROOT/role['pack'],'full_selector_actor')
    for j,cmd in enumerate(raw['actions'][:r['t']+1]):
        obs=public(env,s);a=clean(own.agent(obs,raw['game_configuration']))
        if a!=clean(cmd[s]):raise ValueError(('original full policy prefix differs',name,j))
        if j<r['t']:prefix.append(obs);env.step(cmd);continue
        sys.path.insert(0,str(HERE.parent/'freight_horizon'))
        from predictor import FreightPredictor
        model=HERE.parent/'freight_horizon/model.json'
        assert digest(model.read_bytes())==reg['model_sha256']
        forecast=FreightPredictor(model).predict(obs,prefix[-5:])
        full=choose_intervention(obs,raw['game_configuration'],s,a,forecast)
        if not full or any(r[key]!=full[key] for key in full):raise ValueError('whole policy selector inconsistent')
        save(wd/'root_input.json.gz',dict(public_observation=obs,public_history=prefix[-5:],own_base_actions=a,forecasts=forecast,selection=full))
    case=dict(seed=role['seed'],seat=s,family='fork_denial',root_t=r['t'],source_dir=str((BASE/role['source']).relative_to(ROOT)),
        record=dict(seed=role['seed'],seat=s,source=r['source'],source_sha256=r['source_sha256']),root={**r,'enemy_worker':r['target_worker']},
        pack_sha256=role['sha'],root_list_sha256=digest((HERE/'roots.json').read_bytes()),root_input_sha256=digest((wd/'root_input.json.gz').read_bytes()),
        responder_sha256=digest(Path(live.__file__).read_bytes()),plan_sha256=reg['protocol_sha256'],model_sha256=reg['model_sha256'])
    plan=wd/'live_protocol.json'
    if plan.exists() and read(plan)!=case:raise ValueError('changed root selection')
    save(plan,case);live.prepare(wd)
def arm(name,label):
    reg,ci,r,role,wd=cfg(name);case=read(wd/'live_protocol.json');live.PACK=ROOT/role['pack'];live.PACK_HASH=role['sha']
    assert set(os.sched_getaffinity(0))=={2} and digest(Path(live.__file__).read_bytes())==case['responder_sha256']
    live.run(label,wd,case,resume=True)
def verify(name):
    reg,ci,r,role,wd=cfg(name);raw=read(ROOT/r['source']);s=r['seat'];op=1-s;r={**r,'enemy_worker':r['target_worker']}
    helper=BASE/'planter_routes_dev/endgame_responses';sys.path.insert(0,str(helper))
    spec=importlib.util.spec_from_file_location('official_cash_attribution_freight',helper/'verify.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    arms={};tapes={};control_input=read(wd/'root_input.json.gz');hist=raw['actions'][r['t']]
    for label in ('parent','alternative'):
        got=read(wd/f'live_{label}.json');path=wd/got['action_tape'];tail=read(path);tapes[label]=tail
        assert digest(path.read_bytes())==got['action_sha256'] and got['pack_sha256']==role['sha'] and got['source_sha256']==live.MASTER_SHA and got['shared_init_sha256']==digest((wd/'live_shared_init.json').read_bytes())
        same=clean(tail[0][op])==clean(hist[op]);a,b=clean(hist[s]),clean(tail[0][s]);diff={u for u in a.keys()|b.keys() if a.get(u)!=b.get(u)}
        assert same and diff==({r['worker']} if label=='alternative' else set())
        arm=mod.ledger(raw,r,tail);assert arm['score_by_physical_seat']==got['score'] and len(tail)-1==got['search_calls']
        # Compare with full controller state initialized from the original opening, not a zeroed root.
        own=live.module(ROOT/role['pack'],'full_reactive_actor_'+label);env=begin(raw)
        for j,cmd in enumerate(raw['actions'][:r['t']]+tail):
            inferred=clean(own.agent(public(env,s),raw['game_configuration']))
            if inferred!=clean(hist[s] if j==r['t'] else cmd[s]):raise ValueError(('full actor differs',name,label,j))
            env.step(cmd)
        arm.update(search_calls=got['search_calls'],searched=got['search_completed'],aborted=got['search_aborted'],single=got['search_single'],
                  complete_frozen_actor_replay_steps=env.states[0].observation.step)
        arms[label]=arm
    pa,al=arms.values();channels={k:round(al['gross'][s][k]-al['gross'][op][k]-pa['gross'][s][k]+pa['gross'][op][k],4) for k in pa['gross'][0]}
    d=round(al['margin']-pa['margin'],2);assert abs(sum(channels.values())-d)<.022
    out=dict(root=r,role=name,arms=arms,delta_margin=d,delta_u=al['u']-pa['u'],delta_target_banked=round(al['focal_enemy_deposits']-pa['focal_enemy_deposits'],4),
             target_receipts={k:[x for x in v['deposits'] if x['worker']==r['enemy_worker'] and x['seat']==op] for k,v in arms.items()},
             root_cash_captured_share="Target deposits include all of that same worker's actual subsequent deposits to its own bank; they are kept separate from the model's currently-held-cargo label.",
             opponent_order_different_at=[r['t']+i for i,(a,b) in enumerate(zip(*tapes.values())) if clean(a[op])!=clean(b[op])],sources=channels,
             ledger_sha256=digest((helper/'verify.py').read_bytes()))
    save(wd/'verify.json.gz',out);print(name,d,out['delta_target_banked'],flush=True)
if __name__=='__main__':
    import subprocess
    if not len(sys.argv)>1:
        for role in PLAN['roles']:
            name=role['name']
            if cfg(name):prepare(name)
        for role in PLAN['roles']:
            name=role['name']
            if not cfg(name):continue
            for arm_name in ('parent','alternative'):subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),name,arm_name],check=True)
            verify(name)
    else:arm(*sys.argv[1:3])

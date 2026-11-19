"""Run the public frontend once per root. Separately resolve the signed simultaneous original action."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time
from build import ACTOR,HERE,ROOT,build,module,sha

ev=module(ACTOR/'endgame_queue_dev/study.py','signed_carbon_accounts')
OP=ACTOR/'opening_pair_dev/games/candidate'

def serial(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def physical(cfg,state,joint,seat):
    env=ev.FastCarbon(cfg)
    env.set_state(state)
    rec=ev.twin_step(env,joint)
    return {'status':rec['status'],'score':rec['score'],'cash':rec['cash'],'flow':rec['flow'],'deaths':rec['deaths'],
            'carried':rec['inventory'],'margin':rec['score'][seat]-rec['score'][1-seat],'collisions':rec['collisions']}

def game(actor,seed,seat,protocol):
    receipt=OP/f'receipt_{seed}_{seat}.json'
    signed=ev.read(receipt)
    row=signed['game']
    raw=OP/row['raw_file']
    tape=ev.read(raw)
    assert ev.sha(raw)==row['raw_sha256'] and signed['binding']['candidate']==protocol['parent']==tape['pack_sha256']
    assert signed['official']['statuses']==['DONE','DONE'] and signed['official']['steps']==len(tape['actions'])==299
    cfg=tape['game_configuration'];env=ev.FastCarbon(cfg).reset(seed=seed)
    assert ev.digest(bytes(map(int,env.carbon)))==signed['official']['map_sha256']
    result=[]
    for t,joint in enumerate(tape['actions']):
        obs=env.obs(seat)
        assert 'full_carbon' not in obs and obs['step']==t
        if t in (0,278) or t>=279:
            start=time.process_time()
            selected=actor.agent(obs,cfg)
            ms=(time.process_time()-start)*1000
            actual,trace=actor._FRONT_AUDIT[2:]
            assert actual==joint[seat]
            if t>=279:
                state=env.get_state()
                alt=list(joint);alt[seat]=selected
                baseline=physical(cfg,state,joint,seat)
                changed=actual!=selected
                other=physical(cfg,state,alt,seat) if changed else baseline
                for k,arm in enumerate(trace['arms']):
                    same=list(joint);same[seat]=arm['command']
                    measured=physical(cfg,state,same,seat) if k else baseline
                    arm['held_fixed_root_enemy_one_step']=measured
                    arm['actual_one_step_margin_gain']=measured['margin']-baseline['margin']
                result.append(dict(t=t,H=299-t,public_sha256=sha(serial(obs)),trace=trace,base=actual,selected=selected,
                                   changed=changed,cpu_ms=ms,root_simultaneous_opponent=joint[1-seat],
                                   observed_parent_ledger=baseline,one_step_selected_ledger=other,
                                   actual_paired_step_margin_delta=other['margin']-baseline['margin'],
                                   exact_terminal_delta=(other['margin']-baseline['margin']) if other['status']==baseline['status']==['DONE','DONE'] else None))
        else:
            actor._FEAT.observe(obs)
        env.step(*joint)
    assert sha(serial(env.get_state()))==row['final_state_sha256']
    return dict(seed=seed,seat=seat,source=str(raw.relative_to(ROOT)),terrain=signed['official']['map_sha256'],raw_sha256=ev.sha(raw),
                receipt_sha256=ev.sha(receipt),observed_official_scores=signed['official']['official_reward_by_seat'],rows=result)

def summarize(games):
    rows=[z for g in games for z in g['rows']]
    changed=[r for r in rows if r['changed']]
    total=sum(r['actual_paired_step_margin_delta'] for r in rows)
    return {'games':len(games),'roots':len(rows),'with_pay_or_block':sum(bool(r['trace']['roots']) for r in rows),
        'pay_intent_roots':sum(any(x['pay_intent'] for x in r['trace']['roots']) for r in rows),
        'potential_block_roots':sum(any(x['cargo_corridor'] for x in r['trace']['roots']) for r in rows),
        'legal_comparisons':sum(len(r['trace']['arms'])>1 for r in rows),'changed':len(changed),'mean_step_gain_all':total/len(rows),
        'changed_one_step_deltas':[round(r['actual_paired_step_margin_delta'],2) for r in changed],
        'changed_projected_gains':[r['trace']['arms'][r['trace']['chosen']]['gain'] for r in changed],
        'changed_H1':sum(r['H']==1 for r in changed),'cpu_ms_median':statistics.median(r['cpu_ms'] for r in rows),
        'cpu_ms_max':max(r['cpu_ms'] for r in rows)}

if __name__=='__main__':
    candidate=build()
    p=ev.read(HERE/'protocol.json');assert sha((HERE/'protocol.json').read_bytes())=='144dd5c766425df53da2e6fb0c3bc381bfe7f1c435ad182252d45f52f653584e'
    a=module(candidate,'standalone_sapling_frontier')
    gs=[]
    for seed in (p['diagnostic_seed'],p['held_seed']):
        for seat in (0,1):
            gs.append(game(a,seed,seat,p))
            print(seed,seat,summarize(gs[-1:]),flush=True)
    result={'source_sha256':{str(f.relative_to(ROOT)):ev.sha(f) for f in (HERE/'protocol.json',HERE/'runtime.py',HERE/'build.py',HERE/'audit.py',HERE/'package.json',Path(ev.__file__))},
            'games':gs,'result':{'development':summarize(gs[:2]),'held':summarize(gs[2:]),'all':summarize(gs)}}
    ev.write(HERE/'diagnosis.json.gz',result)

"""Independently replay the eight sealed terminal responses in zerosum_env without invoking the examiner."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np

from study import HERE, ROOT, ENGINE, FastCarbon, CashEvents, read, sha, write, require
from zerosum_env import make

PROTOCOL = '851a0903ab1aa0898ecaf6799528981872358fc14856905742c8a913deef3d04'
RUNNER = '5c135876f0324c694bcb9f71de63c13c25762f88cab088e662231a2cc29eed1c'
SUPPLEMENT = HERE.parent / 'tree_response_training/supplement_01'


def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def clean(commands):
    return {u:c for u,c in commands.items() if c and c!='NONE'}


def utility(score, seat):
    margin=score[seat]-score[1-seat]
    outcome=1. if margin>0 else 0. if margin<0 else .5
    return .5*outcome+.5/(1+math.exp(-margin/1000))


def public(env, seat):
    full=env.states[0].observation
    view=env.states[seat].observation
    return dict(player=seat,step=full.step,players=full.players,trees=full.trees,carbon=view.carbon,
                remainingOverageTime=view.remainingOverageTime)


def match(env, sim):
    official=env.states[0].observation
    require(official.step==sim.step_count and list(official.full_carbon)==sim.carbon and
            all(public(env,p)==sim.obs(p) for p in (0,1)), f'full/filtered state differs at {sim.step_count}')


def advance(env, sim, joint, receipt=None):
    before=sim.full_observation()
    require(not env.done,'actions after terminal')
    for p,commands in enumerate(joint):
        require(set(commands)<=set(before['players'][p][1])|set(before['players'][p][2]), 'foreign order')
    env.step(joint)
    sim.cash=CashEvents(sim.cash) if receipt else list(sim.cash)
    sim.step(*joint)
    match(env,sim)
    if receipt is None:
        return
    after=sim.full_observation()
    flow=[Counter(),Counter()]
    for key,seat,amount,*_ in sim.cash.events:
        flow[seat][key]+=amount
    flows=[dict(g) for g in flow]
    status=[a.status for a in env.states];score=[a.reward for a in env.states]
    dead=[sorted(set(a[2])-set(b[2])) for a,b in zip(before['players'],after['players'])]
    inventory=[sum(w[1] for w in p[2].values()) for p in after['players']]
    require(flows==receipt['flow'] and list(sim.cash.events)==receipt['events'] and dead==receipt['deaths'] and
            inventory==receipt['inventory'] and [list(r) for r in sim.last_collisions]==receipt['collisions'], 'income/worker evidence differs')
    require(receipt['cash']==list(sim.cash) and score==receipt['score'] and status==receipt['status'],'independent official cash/reward')
    require(all(abs(before['players'][p][0]+sum(flows[p].values())-sim.cash[p])<1e-6 for p in (0,1)), 'flow balance')
    return {'step':sim.step_count,'cash_before':[p[0] for p in before['players']],'cash_after':list(sim.cash),
            'official_score':score,'status':status,'flows_by_seat':flows,'physical_bank_receipts':[
                e for e in sim.cash.events if e[0] in ('own_delivery','opponent_worker_delivery_to_this_base','base_collision_cash')],
            'worker_deaths':dead,'carried_inventory':inventory}


def verify_arm(pair, key, name, expected, stream, plan, raw, package, decision, reply):
    seed,seat=key;enemy=1-seat
    leaf=HERE/f'last_arm_{seed}_{seat}_{name}.json.gz'
    checkpoint=read(leaf); signed=pair['arms'][name]
    require(checkpoint['bindings']==[PROTOCOL,expected['source_sha256'],RUNNER,name] and checkpoint['outcome']==signed,'arm seal')
    require(signed['rng_before_sha256']==pair['paired_rng_state_sha256']==canonical(stream) and
            signed['cost_before']==plan['cost_before_each_arm'] and signed['search_callback_seat_step']==[[enemy,298]]
            and signed['search_call_count']==1 and signed['native_action_wall_s']>0, 'native callback and start-state attestation')
    cfg=raw['game_configuration']; np.random.seed(seed)
    env=make('carbon',configuration={**cfg,'randomSeed':seed})
    require(list(env.states[0].observation.full_carbon)==raw['initial_full_carbon'] and
            hashlib.sha256(bytes(env.states[0].observation.full_carbon)).hexdigest()==raw['initial_terrain_u8_sha256'],'official initial terrain')
    sim=FastCarbon(cfg).reset(full_carbon=raw['initial_full_carbon'])
    match(env,sim)
    history=hashlib.sha256(); official_history=hashlib.sha256()
    runtime_cfg=dict(cfg,randomSeed=None)
    for t in range(298):
        joint=raw['actions'][t]
        their=public(env,enemy); own=public(env,seat)
        # Equivalent numeric data can serialize differently in the two engines (e.g. 0 and 0.0).
        history.update(canonical((sim.obs(enemy),joint[enemy])).encode())
        official_history.update(canonical((their,joint[enemy])).encode())
        if t==0:
            require(clean(package.agent(own,runtime_cfg))==clean(joint[seat]),'opening policy reproduction')
        elif t<297:
            package._FEAT.observe(own)
            package._queue_remember(own)
        else:
            break
        advance(env,sim,joint)
    require(history.hexdigest()==pair['public_history_sha256'] and pair['public_history_turns']==298, 'enemy prefix visibility/action history')
    offer=clean(package.agent(own,runtime_cfg)); original=clean(package._QUEUE_TRACE[2]); enemy_first=raw['actions'][297][enemy]
    require(original==clean(raw['actions'][297][seat]) and offer==clean(decision['candidate_own']) and
            decision['changes']==expected['edits'] and canonical(sim.obs(seat))==signed['own_root_visible_sha256'], 'root public decision')
    diff={k:[original.get(k),offer.get(k)] for k in sorted(original.keys()|offer.keys()) if original.get(k)!=offer.get(k)}
    require(diff==expected['edits'], 'root treatment or timing differs')
    first=signed['first_joint']; last=signed['last_joint']
    require(first[enemy]==enemy_first and clean(first[seat])==(original if name=='v1' else offer) and
            signed['first_joint_sha256']==canonical(first) and signed['last_joint_sha256']==canonical(last), 'joint action seals')
    # The signed runner restored its FastCarbon snapshot before each fork. This normalizes
    # scalar types for the byte-level public-observation digest; the official environment
    # continues from turn zero, and every restored physical and filtered value must match it.
    sim.set_state(sim.get_state())
    match(env,sim)
    step1=advance(env,sim,first,signed['first'])
    require(step1['status']==['ACTIVE','ACTIVE'] and not env.done and sim.step_count==298,'first turn terminal')
    require(canonical(sim.obs(seat))==signed['own_last_visible_sha256'] and
            canonical(sim.obs(enemy))==signed['enemy_last_visible_sha256'], 'independent forked observations')
    choice=clean(package.agent(public(env,seat),runtime_cfg)); own_v1=clean(package._QUEUE_TRACE[2])
    expected_last=next(r for r in reply['arms'] if r['arm']==name)['v1_last_commands']
    require(own_v1==clean(expected_last) and clean(last[seat])==(own_v1 if name=='v1' else choice), 'last Sapling order')
    if name=='v1':
        require(own_v1==clean(raw['actions'][298][seat]), 'original v1 final order')
    else:
        require(choice==own_v1, 'unexpected second intervention')
    require(signed['enemy_matches_original_recorded_last']==(clean(last[enemy])==clean(raw['actions'][298][enemy])), 'enemy contrast')
    step2=advance(env,sim,last,signed['end'])
    require(env.done and step2['status']==['DONE','DONE'] and sim.step_count==299,'missing official end')
    stats=signed['native_search']; completed=stats['searched']; single=stats['single']; aborted=stats['aborted']
    require(completed in (0,1) and single in (0,1) and aborted in (0,1) and completed+single<=1
            and completed+single+aborted>=1 and len(stats['time'])==completed and len(signed['root_candidates_reported'])==completed,
            'original search counters inconsistent')
    for step,player,candidates,depths,values,best in signed['root_candidates_reported']:
        require((step,player)==(298,enemy) and len(candidates)==len(depths)==len(values) and 0<=best<len(candidates)
                and all(math.isfinite(v) for v in values) and clean(candidates[best])==clean(last[enemy]), 'selected searched candidate vs actual enemy order')
    require(all(math.isfinite(v) and v>0 for v in signed['cost_after']) and len(signed['cost_after'])==len(signed['cost_before']), 'invalid timing state')
    return {'arm_sha256':sha(leaf),'source_prefix_turns':297,'history_sha256':history.hexdigest(),
            'independent_official_history_sha256':official_history.hexdigest(),'official_terminal':step2['official_score'],
            'first':step1,'last':step2,'first_joint_sha256':canonical(first),'last_joint_sha256':canonical(last),
            'enemy_matches_original_last':signed['enemy_matches_original_recorded_last'],'changed_final_own_from_source':
            clean(last[seat])!=clean(raw['actions'][298][seat]),'examiner_public_last_sha256':signed['enemy_last_visible_sha256'],
            'initial_rng_sha256':signed['rng_before_sha256'],'initial_cost':signed['cost_before'], 'native_search':{
            'callbacks':1,'seat_step':signed['search_callback_seat_step'][0],'completed_search':completed,'single_action':single,
            'aborted':aborted,'policy_choice_changed':stats['changed'],'depth':stats['depth'],'rollouts':stats['rolls'],
            'candidate_counts':stats['cands'],'wall_seconds':signed['native_action_wall_s']}}


def report():
    plan=read(HERE/'native_protocol.json'); total=read(HERE/'native_result.json')
    require(sha(HERE/'native_protocol.json')==PROTOCOL==total['protocol_sha256'] and
            sha(HERE/'native_last.py')==RUNNER==total['runner_sha256'],'frozen native runner')
    for name,field in {'replay_roots.json.gz':'sources_sha256','root_audit.json.gz':'raw_evidence_sha256',
            'protocol.json':'upstream_protocol_sha256','v1_last_reply.json.gz':'parent_reply_sha256','policy.py':'policy_sha256',
            'candidate.py':'candidate_sha256','native_streams.json.gz':'shared_streams_sha256'}.items():
        require(sha(HERE/name)==plan[field],f'upstream {name}')
    grant_path=SUPPLEMENT/'endgame_queue_grant.json';complete=SUPPLEMENT/'completion.json'; grant=read(grant_path)
    require(sha(grant_path)==total['grant_sha256'] and sha(complete)==grant['completion_sha256']==total['completion_sha256']
            and grant['cpu']==0 and grant['permitted_native_calls']==total['native_calls']==8 and total['cpu_affinity']==[0], 'slot and callback authorization')
    require(sha(ROOT/plan['cost_history'])==plan['cost_history_sha256'] and read(ROOT/plan['cost_history'])['warm_cost']==plan['cost_before_each_arm'], 'frozen calibration')
    require(total['candidate_sha256']==plan['candidate_sha256'] and sha(ROOT/'data/master3s2_source.py')==plan['source_examiner']==
            hashlib.sha256(subprocess.check_output(['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=ROOT)).hexdigest()==total['examiner_sha256'], 'agent and examiner identity')
    require(total['source_params']=={'temperature':1,'seed':None,'search':True,'cost':plan['cost_before_each_arm'],'common_rng':True}, 'search settings')
    module=importlib.util.spec_from_file_location('independent_queue_package',HERE/'candidate.py')
    package=importlib.util.module_from_spec(module); module.loader.exec_module(package)
    original=read(HERE/'protocol.json'); sources={};engine_files={}
    for b in original['source_batches']:
        if b['role']!='evaluation':continue
        path=ROOT/b['receipt']; reg=read(path); replay_path=path.parent/'source/official_replay.json'; off=read(replay_path)
        require(sha(path)==b['receipt_sha256'] and sha(replay_path)==b['official_sha256']==reg['official_sha256'], 'source official binding')
        for name,value in off['engine_source_sha256'].items():
            require(value==sha(ENGINE/name),'official engine code')
            engine_files[name]=value
        for game in reg['games']:
            key=game['seed'],game['seat']; sig,=[x for x in off['runs']['source']['games'] if (x['seed'],x['seat'])==key]
            require(key not in sources and game['scores']==sig['official_reward_by_seat'] and sig['statuses']==['DONE','DONE'] and
                    sig['transcript_sha256']==game['raw_sha256'] and sig['map_sha256']==game['map_sha256'], 'signed source membership')
            sources[key]=game
    cases={(c['seed'],c['seat']):c for c in plan['cases']}
    votes={(r['seed'],r['seat'],r['t']):r for r in read(HERE/'root_audit.json.gz') if r['split']=='evaluation'}
    replies={(r['seed'],r['seat']):r for r in read(HERE/'v1_last_reply.json.gz')['roots']}
    streams=read(HERE/'native_streams.json.gz'); signed={(r['seed'],r['seat']):r for r in total['pairs']}
    require(len(streams)==len(cases)==len(signed)==len(total['pair_sha256'])==4 and len(sources)==16 and len(votes)==32 and
            all((k in cases)==votes[*k,297]['changed'] and not votes[*k,298]['changed'] for k in sources),'registered all-source denominator')
    outcomes={}; per_root=[]
    for i,(key,expected) in enumerate(cases.items()):
        seed,seat=key;enemy=1-seat; source=sources[key]; raw_path=ROOT/source['raw_path']; raw=read(raw_path)
        path=HERE/f'native_{seed}_{seat}.json.gz'; pair=read(path); stream=streams[i]
        require(sha(path)==total['pair_sha256'][path.name] and sha(raw_path)==source['raw_sha256']==expected['source_sha256']==pair['source_sha256'] and
                pair['protocol_sha256']==PROTOCOL and pair['runner_sha256']==RUNNER and raw['official_score_by_seat']==pair['original_scores_physical']==source['scores'], 'paired envelope')
        require((stream['seed'],stream['seat'])==key and (pair['seed'],pair['seat'])==key and
                list(pair['arms'])==(['v1','candidate'] if i%2==0 else ['candidate','v1']), 'paired stream and order')
        arms={name:verify_arm(pair,key,name,expected,stream,plan,raw,package,votes[*key,297],replies[key]) for name in pair['arms']}
        old,new=[arms[k]['official_terminal'] for k in ('v1','candidate')];delta=[round(y-x,2) for x,y in zip(old,new)]
        margin=round(delta[seat]-delta[enemy],2);delta_u=utility(new,seat)-utility(old,seat)
        enemy_changed=pair['arms']['v1']['last_joint'][enemy]!=pair['arms']['candidate']['last_joint'][enemy]
        require(pair['native_enemy_orders_differ']==enemy_changed and pair['delta_rewards_physical']==delta and pair['delta_margin']==margin
                and math.isclose(pair['delta_u'],delta_u,abs_tol=1e-12) and all(pair[k]==v for k,v in signed[key].items()),'terminal contrast')
        require(pair['won_by_arm']=={k:r['official_terminal'][seat]>r['official_terminal'][enemy] for k,r in arms.items()},'win flags')
        outcomes[key]=(margin,delta_u)
        per_root.append({'seed':seed,'seat':seat,'terrain':source['map_sha256'],'pair_sha256':sha(path),'arms':arms,
                         'recorded_terminal_rewards':source['scores'],'delta_physical':delta,'delta_margin':margin,'delta_utility':delta_u,
                         'enemy_last_differs_between_arms':enemy_changed})
        print(f'official terminal verified: {seed} / {seat}',flush=True)
    denominators=[]
    for (seed,seat),src in sorted(sources.items()):
        margin,u=outcomes.get((seed,seat),(0.,0.))
        denominators.append({'seed':seed,'terrain':src['map_sha256'],'seat':seat,'evaluated_native_pair':(seed,seat) in outcomes,
                             'delta_margin':margin,'delta_utility':u})
    per_map=defaultdict(list)
    for row in denominators:per_map[row['terrain']].append(row)
    require(len(per_map)==8 and all({r['seat'] for r in rr}=={0,1} and len(rr)==2 for rr in per_map.values()),'two seats per evaluated terrain')
    def aggregate(rows):
        return {'games':len(rows),'native_pairs':sum(r['evaluated_native_pair'] for r in rows),
                'mean_delta_margin':mean(r['delta_margin'] for r in rows),'mean_delta_utility':mean(r['delta_utility'] for r in rows)}
    map_means={s:aggregate(v) for s,v in per_map.items()}; all_games=aggregate(denominators)
    require(math.isclose(mean(r['mean_delta_margin'] for r in map_means.values()),total['mean_delta_all_16'],abs_tol=1e-10) and
            math.isclose(mean(r['mean_delta_utility'] for r in map_means.values()),total['mean_delta_u_all_16'],abs_tol=1e-12), 'equal-map summary')
    result={'schema':'independent-endgame-native-official-replay-v1','native_protocol_sha256':PROTOCOL,'native_result_sha256':sha(HERE/'native_result.json'),
            'verifier_sha256':sha(Path(__file__)),'engine_sha256':engine_files,'registered_native_calls':8,'completed_native_pairs':4,
            'whole_trajectories_replayed_in_official_env':8,'original_examiner_calls_in_verifier':0,'whole_game_native_matches_started':0,
            'paired_allocation':denominators,'all_sixteen_physical_games':all_games,'per_map':map_means,
            'per_focal_seat':{p:aggregate([r for r in denominators if r['seat']==p]) for p in (0,1)},'pairs':per_root,
            'unfired_twelve':'Deterministic zero contrast for twelve original paths with no root or last-turn edit; no native callbacks on these paths.',
            'native_scope':'Per-arm callbacks, starting RNG and calibrated costs are checked against sealed readbacks; native wall-clock searches are not rerun.'}
    write(HERE/'native_official.json',result)
    print(json.dumps(all_games))


if __name__=='__main__':report()

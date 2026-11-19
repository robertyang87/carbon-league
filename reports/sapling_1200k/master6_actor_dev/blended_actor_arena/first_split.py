"""After each signed pair, attribute exactly one official transition at its last guaranteed shared prefix."""
import sys
from collections import Counter
from pathlib import Path
from shared import ACTOR,HERE,ROOT,module,read,require,save,sha

TWIN=ACTOR/'endgame_queue_dev/study.py'


def analyze(arms,seed,seat,twin):
    j=[v['actions'] for v in arms];n=min(map(len,j))
    def first(which):return next((i for i in range(n) if twin.clean(j[0][i][which])!=twin.clean(j[1][i][which])),None)
    own,enemy=first(seat),first(1-seat);both=[k for k in (own,enemy) if k is not None]
    require(both,'entire current pair has identical normalized actions')
    t=min(both);labels=('parent','candidate');cells={};diff={}
    for actor_index in (0,1):
        for opp_index in (0,1):
            sim=twin.FastCarbon(arms[0]['game_configuration']).reset(seed=seed)
            for action in j[0][:t]:sim.step(*action)
            sim.cash=twin.CashEvents(list(sim.cash))
            oa=j[actor_index][t][seat];ea=j[opp_index][t][1-seat]
            pair=[None,None];pair[seat]=oa;pair[1-seat]=ea
            key=f'{labels[actor_index]}_own__{labels[opp_index]}_opponent'
            cells[key]=twin.twin_step(sim,pair)
            cells[key]['joint_commands']=pair
            cells[key]['factual_for']=([labels[actor_index]] if actor_index==opp_index else [])
    for b in labels:diff[b]=twin.difference(cells['parent_own__'+b+'_opponent'],cells['candidate_own__'+b+'_opponent'],seat)
    return dict(seed=seed,seat=seat,terminal_official_margin_delta=arms[1]['margin']-arms[0]['margin'],
                first_own_command_difference=own,first_opponent_command_difference=enemy,first_joint_difference=t,
                first_changes='both' if own==enemy else ('own' if t==own else 'opponent'),settlements=cells,
                one_transition_own_action_effect_at_fixed_opponent=diff)


def opening_composition(raw,seat,twin):
    """Descriptive full trajectories, evaluated for all eight, after observing one first-slot difference."""
    sim=twin.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    sim.cash=twin.CashEvents(list(sim.cash));first={};n=[Counter(),Counter()];cash=[Counter(),Counter()];out=[]
    for t,pair in enumerate(raw['actions'][:100]):
        sim.cash.events.clear();sim.step(*pair)
        for code,who,delta,*_ in sim.cash.events:
            cash[who][code]+=delta;n[who][code]+=1
            if code in ('hire_planter','hire_collector','plant','seize','own_delivery'):
                first.setdefault(str(who)+'/'+code,t)
        if t+1 in (1,5,10,20,50,100):
            obs=sim.obs(seat);states=[]
            for side,pl in enumerate(obs['players']):
                worker=Counter(k for _,_,k in pl[2].values())
                states.append(dict(workers=dict(worker),trees=len(pl[3]),cash=pl[0],
                                   own_deposits=cash[side]['own_delivery'],tree_absorption=cash[side]['tree_absorption'],
                                   collectors_recruited=n[side]['hire_collector'],planters_recruited=n[side]['hire_planter']))
            out.append(dict(completed_turns=t+1,physical_seats=states))
    return dict(first_real_event_step_by_seat=first,snapshots=out)

def run(partial=False):
    twin=module(TWIN,'blend_one_turn_official');out=[];binding=read(HERE/'games/binding.json')
    for s in binding['boards']:
        for seat in (0,1):
            paths=[HERE/'games'/a/f'raw_{s}_{seat}.json.gz' for a in ('parent','candidate')]
            if partial and not all(p.exists() and p.with_name(p.name.replace('raw_','receipt_').replace('.json.gz','.json')).exists() for p in paths):continue
            arms=[read(p) for p in paths];require(arms[0]['initial_terrain_u8_sha256']==arms[1]['initial_terrain_u8_sha256']==binding['boards'][s],'different starting boards')
            require(all(b['pack_sha256']==binding[a] for a,b in zip(('parent','candidate'),arms)),'pack')
            row=analyze(arms,int(s),seat,twin);row['raw_sha256']=[sha(p) for p in paths];row['actual_opening_composition']=dict(zip(('parent','candidate'),[opening_composition(z,seat,twin) for z in arms]));out.append(row)
    result={'complete':len(out)==4,'one_transition_only':True,'counterfactuals_only_at_first_physical_divergence':True,
        'raw_official_comparison_sha256':None if partial else sha(HERE/'comparison.json'),'registry_sha256':sha(HERE/'protocol.json'),
        'twin_engine_sha256':sha(TWIN),'cases':out,'first_side_counts':dict(Counter(r['first_changes'] for r in out))}
    if not partial:require(len(out)==4,'missing results');save(HERE/'first_split.json.gz',result)
    return result


if __name__=='__main__':
    result=run('--partial' in sys.argv);print([(r['seed'],r['seat'],r['first_changes'],r['first_own_command_difference'],r['first_opponent_command_difference']) for r in result['cases']])

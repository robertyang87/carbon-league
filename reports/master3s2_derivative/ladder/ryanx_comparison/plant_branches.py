"""Replay every deficit-flagged v4 plant and ex ante price-matched comparison under the actual opponent commands."""
import copy
import gzip
import json
from pathlib import Path
import sys

from tree_forecast import HERE, FAMILY, LADDER, read, sha
sys.path.insert(0, str(FAMILY.parents[1] / 'code/env_src/carbon_challenge-master'))
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.helpers import Board

OPTIONS = ('NONE','UP','RIGHT','DOWN','LEFT')


def main():
    scored = read(HERE/'tree_forecast.json')['games'][1]
    game_file = LADDER/'v4_observations/replays'/f"{scored['match']}-1.json.gz"
    assert sha(game_file) == scored['replay_sha256']
    replay = read(game_file)
    frames = replay['steps']
    cfg = replay['configuration']
    team = scored['teams'][0]
    seat = team['seat']
    plants = team['rows']
    negative = [r for r in plants if r['predicted_net'] < 0]
    nonnegative = [r for r in plants if r['predicted_net'] >= 0]
    controls = []
    for r in negative:
        control = min(nonnegative, key=lambda c: (abs(c['cost']-r['cost']), abs(c['step']-r['step']), c['step'], c['tree']))
        controls.append(control)
        nonnegative.remove(control)
    targets = negative + controls
    cfg_size = cfg['size']
    def commands(t):
        return [dict(side['action'] or {}) for side in frames[t+1]]
    def history(t):
        obs = copy.deepcopy(frames[t][0]['observation'])
        obs['carbon'] = obs['full_carbon']
        return obs
    def advance(state, joint):
        return Board(state,cfg,joint).next().observation
    def margin(state):
        players=state['players']
        return players[seat][0]-players[1-seat][0]
    roots = {}
    idgen.reset()
    for t in range(len(frames)-1):
        state = history(t)
        if any(r['step']==t for r in targets):
            roots[t] = dict(idgen._id_dict)
        post=advance(state,commands(t))
        recorded=history(t+1)
        assert all(post[k]==recorded[k] for k in ('players','trees','carbon','step'))
    findings=[]
    for r in targets:
        t=r['step']
        uid=r['planter']
        results={}
        assert commands(t)[seat].get(uid,'NONE')=='NONE'
        limit=min(10,len(frames)-1-t)
        for action in OPTIONS:
            state=history(t)
            idgen._id_dict.clear()
            idgen._id_dict.update(roots[t])
            trace=[]
            for step in range(t,t+limit):
                joint=commands(step)
                if step==t and action!='NONE':
                    joint[seat][uid]=action
                state=advance(state,joint)
                if step-t+1 in (1,3,5,10):
                    actual=history(step+1)
                    if action=='NONE':
                        assert all(state[k]==actual[k] for k in ('players','trees','carbon','step'))
                    trace.append(dict(offset=step-t+1, delta=round(margin(state)-margin(actual),2),
                                      planter_alive=uid in state['players'][seat][2]))
            results[action]=trace
        findings.append(dict(group='predicted_negative' if r in negative else 'price_matched_nonnegative',
                             step=t,uid=uid,cell=r['cell'],quoted_cost=r['cost'],cell_xy=[r['cell']%cfg_size,r['cell']//cfg_size],
                             predicted_first_tenure_net=r['predicted_net'],observed_first_tenure_net=r['actual_net'],outcomes=results))
    report=dict(source=scored['match'], observed_transitions_checked=len(frames)-1,
                selection='Every v4 acquired new tree with predicted negative first tenure, plus distinct nonnegative predictions matched on price, time and tree identity. All four move-away choices preserved, with controls staying.',
                interpretation='The only change is the one planter first instruction. Later commands of both agents are the recorded strings. After an intervention, the agents did not observe the branch. Score at +10 is cash on hand and does not value remaining assets.',
                cases=findings)
    (HERE/'plant_branches.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for row in findings:
        print(row['group'],row['step'],row['cell'],round(row['observed_first_tenure_net'],2),[(a,tr[-1]['delta']) for a,tr in row['outcomes'].items() if a!='NONE'])


if __name__=='__main__':
    main()

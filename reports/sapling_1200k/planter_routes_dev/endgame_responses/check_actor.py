"""Prove late-actor replay agrees with full frozen-controller history on these paths."""
from pathlib import Path
import gzip,json,sys
from select_roots import HERE,BASE,tape,read,write
from run import configure,live,PACK
from branches import begin,public,clean
def main():
    configure();rw=json.loads(gzip.decompress(tape.read_bytes()));o=read(HERE/'roots.json');env=begin(rw);obs=[]
    for cmd in rw['actions']:
        visible=public(env,1);obs.append(visible);env.step(cmd)
    candidate=live.module(PACK,'full_candidate_original_history')
    for t,(joint,v) in enumerate(zip(rw['actions'],obs)):
        if clean(candidate.agent(v,rw['game_configuration']))!=clean(joint[1]):raise ValueError(('original actor',t))
    records=[]
    for family,r in o['selected'].items():
        t=r['t']
        for rep in range(4):
            for name in ('parent','alternative'):
                wd=HERE/'arms'/f'{family}_{rep}';arm=read(wd/f'live_{name}.json');tail=json.load(gzip.open(wd/arm['action_tape']))
                env=begin(rw);own=live.module(PACK,'full_candidate_alternate_history')
                for j,order in enumerate(rw['actions'][:t]+tail):
                    # All prefix actions were validated against the real replay above. This
                    # extra pass includes the frozen controller's own cooldown and route state.
                    action=clean(own.agent(public(env,1),rw['game_configuration']))
                    want=clean(rw['actions'][j][1] if j<=t else order[1])
                    if action!=want:raise ValueError((family,rep,name,j))
                    env.step(order)
                records.append(dict(family=family,replicate=rep,arm=name,all_states=env.states[0].observation.step,checked_live_policy_steps_after_root=len(tail)-1))
    write(HERE/'actor_recheck.json',dict(full_original_agent_commands_reproduced=len(rw['actions']),all_branches=records))
if __name__=='__main__':main()

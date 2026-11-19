"""Attribute changes on fixed exposed moments of all actual candidate games, using their real public histories."""
from collections import Counter
import gzip
import json
import sys
from build import ROOT,HERE,PARENT,CAND,digest,out
from mechanism import module,read,FastCarbon


def main():
    result=read(HERE/'comparison.json')
    if not result['complete']:raise ValueError('missing signed native games')
    plan=read(HERE/'protocol.json');schedule=set(plan['mechanism_steps'])
    counted=Counter();rows=[]
    for g in result['games']:
        seed,seat=g['seed'],g['seat']
        path=HERE/'native/candidate'/f'raw_{seed}_{seat}.json.gz'
        if digest(path)!=g['candidate_sha256']:raise ValueError('wrong native tape')
        tape=read(path);game=FastCarbon(tape['game_configuration']).reset(seed=seed)
        agents=[module(p,k,w) for p,k,w in [(CAND,'live',.25),(CAND,'no_student',0),(PARENT,'it50_again',.25)]]
        for t,joint in enumerate(tape['actions']):
            obs=game.obs(seat)
            if t in schedule:
                acts=[a.agent(obs,game.config) for a in agents]
                if acts[0]!=joint[seat]:raise ValueError(f'actual live entry mismatch: {seed}/{seat}/{t}')
                details={}
                for i,k in ((1,'student_removed'),(2,'old_expert_restored')):
                    diffs=[(slot,item['role'],item['action'],agents[i].audited[slot]['action'])
                           for slot,item in agents[0].audited.items() if item!=agents[i].audited[slot]]
                    counted[(str(seat),k,'roots')]+=bool(diffs)
                    for _,role,_,_ in diffs:counted[(str(seat),k,role)]+=1
                    details[k]=diffs
                rows.append(dict(seed=seed,seat=seat,t=t,commands=acts,delta=details))
            else:
                for a in agents:a._FEAT.observe(obs)
            game.step(*joint)
    out(HERE/'attribution.json', dict(source_sha256=digest(HERE/'attribution.py'),comparison_sha256=digest(HERE/'comparison.json'),
            code_sha256=digest(HERE/'mechanism.py'),schedule=sorted(schedule),roots=len(rows),
            by_seat={'/'.join(k):v for k,v in counted.items()}, sampled=rows))
    print('Verified actual candidate commands:',len(rows))


if __name__=='__main__':main()

"""Recompute arrival and own coverage in the official engine for both directions at every approach root."""
from measure import HERE,sr
from guard_tree import _D

def run():
    checks=[]
    for case in sr.load(HERE/'approach.json.gz'):
        raw=sr.load(sr.ROOT/case['source']);q=case['choice'];s=case['seat'];t=case['t']
        for arm in ('base','changed'):
            env=sr.live.begin(raw)
            for cmd in raw['actions'][:t]:env.step(cmd)
            cmd=[dict(x) for x in raw['actions'][t]]
            if arm=='changed':cmd[s][q['uid']]=q['after']
            env.step(cmd);obs=sr.live.public(env,s);own=obs['players'][s][2];foes=obs['players'][1-s][2]
            result=dict(covered=any(v[2]=='COLLECTOR' and _D[v[0],q['cell']]<=1 for v in own.values()),
                        planter_on=q['enemy'] in foes and foes[q['enemy']][0]==q['cell'])
            assert result==case[arm],(case,arm,result)
            checks.append([case['source'],t,arm])
    sr.retain(HERE/'official_approach_checks.json',dict(count=len(checks),checks=checks))
    print(len(checks))
if __name__=='__main__':run()

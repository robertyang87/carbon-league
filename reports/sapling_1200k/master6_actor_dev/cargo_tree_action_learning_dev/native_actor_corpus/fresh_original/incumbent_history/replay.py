"""Feed the archived commands to the official referee. No player is invoked."""
import json
import sys
import zipfile

from recount import HERE, EXP, ROOT, artifact, load, sha, sig


def replay():
    sys.path.insert(0,str(ROOT/'code/env_src/carbon_challenge-master'))
    import numpy as np
    from zerosum_env import make
    plan=load(EXP/'ladder_live/protocol.json')
    for k in ('environment','official_helpers'):
        row=plan['files'][k];assert artifact(ROOT/row['path'])['sha256']==row['sha256']
    games=load(HERE/'history.json')['games'];reports=[]
    template=load(EXP/'direct_original/games/candidate/raw_621758346_1.json.gz')['game_configuration']
    for g in games:
        evidence=g['artifacts'];seed,seat=g['seed'],g['seat']
        if 'raw' in evidence:
            p=ROOT/evidence['raw']['path'];assert artifact(p)==evidence['raw'];raw=load(p)
            cfg=raw['game_configuration'];actions=raw['actions']
        else:
            p=ROOT/evidence['archive']['path'];assert artifact(p)==evidence['archive']
            member=next(k for k in evidence if k.endswith('/turns.jsonl'))
            with zipfile.ZipFile(p) as z: data=z.read(member)
            assert sha(data)==evidence[member]['sha256']
            actions=[[slot['response']['action'] for slot in json.loads(line)['calls']] for line in data.splitlines()]
            cfg={**template,'agentTimeout':20,'actTimeout':1,'runTimeout':1500}
        np.random.seed(seed);env=make('carbon',configuration={**cfg,'randomSeed':seed})
        terrain=sha(bytes(env.states[0].observation.full_carbon))
        assert terrain==g['terrain'] and sig([terrain,actions])==g['trajectory_sha256']
        for t,a in enumerate(actions):
            assert not env.done and env.states[0].observation.step==t
            env.step(a)
        scores=[float(s.reward) for s in env.states]
        assert env.done and all(s.status=='DONE' for s in env.states) and scores==g['official_scores_by_physical_seat']
        reports.append(dict(seed=seed,seat=seat,category=g['category'],steps=len(actions),trajectory_sha256=g['trajectory_sha256'],score_by_physical_seat=scores))
    return dict(history_sha256=sha((HERE/'history.json').read_bytes()),official_transitions=sum(g['steps'] for g in reports),games=reports)


if __name__=='__main__':
    (HERE/'official_replay.json').write_text(json.dumps(replay(),indent=2,sort_keys=True)+'\n')

"""Test canceling a last-turn hire or tree purchase against the simultaneous original reply."""
import copy
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'collector_trade_value_dev'))
from value import sr
from role import _TO,_DIRECTION,_D


def choose(obs,commands):
    own,enemy=obs['players'][obs['player']],obs['players'][1-obs['player']]
    if obs['step']!=298:return commands, []
    # Require a surviving worker elsewhere even if a hire is cancelled.
    def unthreatened(cell):
        return all(_D[cell,w[0]]>1 for w in enemy[2].values())
    def land(w,act):
        return int(_TO[w[0],_DIRECTION.index(act)])
    out=commands.copy()
    changes=[]
    if any(unthreatened(land(w,commands.get(k))) for k,w in own[2].items()):
        for k,pos in own[1].items():
            if k in out and unthreatened(pos):
                changes.append([k,out.pop(k),None])
    trees={v[0] for p in obs['players'] for v in p[3].values()}
    bases=set(own[1].values())|set(enemy[1].values())
    for k,w in own[2].items():
        if w[2]!='PLANTER' or out.get(k) or w[0] in trees|bases or not unthreatened(w[0]):continue
        used={land(v,out.get(u)) for u,v in own[2].items() if k!=u}
        for act,z in zip(_DIRECTION[1:],_TO[w[0],1:]):
            if z not in used|bases and unthreatened(z):
                changes.append([k,None,act]);out[k]=act;break
    return out,changes

def run():
    sources=sr.load(HERE.parent/'harvest_contact_dev/coverage.json.gz')
    for d in ('harvest_contact_dev','midgame_role_value_dev'):
        for branch in ('parent','candidate'):
            for p in sorted((HERE.parent/d/'games'/branch).glob('raw_*.json.gz')):
                raw=sr.load(p);sources.append(dict(seed=raw['seed'],seat=int(p.stem.split('_')[-1].split('.')[0]),source=str(p.relative_to(sr.ROOT))))
    rows=[]
    for g in sources:
        p=sr.ROOT/g['source']; raw=sr.load(p); seat=g['seat'];env=sr.live.begin(raw)
        for joint in raw['actions'][:-1]:env.step(joint)
        assert not env.done
        pair=raw['actions'][-1]; obs=sr.live.public(env,seat)
        changed,edits=choose(obs,pair[seat]); alt=copy.deepcopy(pair);alt[seat]=changed
        if edits:
            other=sr.live.begin(raw)
            for joint in raw['actions'][:-1]:other.step(joint)
            other.step(alt);assert other.done;score=[s.reward for s in other.states]
        env.step(pair); before=[s.reward for s in env.states];assert env.done and before==raw['official_score_by_seat']
        if not edits:score=before
        delta=(score[seat]-score[1-seat])-(before[seat]-before[1-seat])
        rows.append(dict(seed=g['seed'],seat=seat,source=g['source'],sha256=sr.sha(p),edits=edits,official_parent=before,official_changed=score,delta=delta))
    sr.retain(HERE/'scores.json',rows)
    print(len(rows),[(r['seed'],r['seat'],r['edits'],r['delta']) for r in rows if r['edits']])
if __name__=='__main__':run()

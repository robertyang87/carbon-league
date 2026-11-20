"""Freeze independent physical maps, old source turns and scores to compute before forecasting."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_bytes())
def main():
    ref=load(D/'advantage_policy_update/protocol.json')
    tickets=[dict(group='previous_hold',rel=c['rel'],arm=c['arm'],seed=c['seed'],seat=c['seat'],
                  terrain=c['map'],archive_sha256=c['archive_sha256']) for c in ref['tickets'] if c['split']=='held']
    # Include BOTH original seats on every one of the four oldest dev1 maps, even if an archived game timed out.
    old=load(D/'dev1/hosted/settlement.json')
    for c in old['games']:
        z=Path(c['archive_path'])
        tickets.append(dict(group='external',rel='dev1/hosted/'+z.name,arm=c['arm'],seed=c['seed'],seat=c['seat'],
           terrain=c['terrain'],archive_sha256=sha((D/'dev1/hosted'/z.name).read_bytes())))
    groups={g:{v['terrain'] for v in tickets if v['group']==g} for g in ('previous_hold','external')}
    assert all(len(v)==4 for v in groups.values()) and groups['previous_hold'].isdisjoint(groups['external'])
    assert not set.union(*groups.values()).intersection({r['map'] for r in ref['tickets'] if r['split']=='fit'})
    doc=dict(work='v4 root half-turn probability ensemble retrospective',baseline='cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a',candidate='cb64b1065192631f0d03b065bdcdfdb81e93b0c11788cad66cb27c54e8aae271',
      tickets=tickets, steps=ref['steps'], source_selection='Reuse previously declared four hold maps (602–605); independently retain all eight tickets on four prior dev1 maps 580–583. Both seats throughout, no sampling by score or command. Use only complete 299-turn healthy branches; count any incomplete source.',
      prediction='Use frozen 8b4-derived v4 own root worker/base head versus its 180-deg ensemble on same V4 input. Label=actual simultaneously publicly issued order. Primary = archived original Master3s2 side, split baseline from second metric historical source side. Each unique map equally weighted; within map worker/base equally weighted where both eligible; report both counts, seat and separate map/cohort. Eligible bases sequentially at least 30 cash and room under cap 10 as existing AWR; primary workers exclude deterministic parent go_home rule, plus all-issued sensitivity. Unweighted cross entropy and argmax on raw root heads.',
      behavior='Secondary labels only: cross-side same target cell for worker given actual simultaneous joint orders, actual credited own-bank deposit by UID, observed newly planted/seized tree by actual planter. Report full and conditioned counts, LC and decline examples without causal claims. Never input future observations, outcome or opposing command.',
      cohorts='Previous hold reused for architecture review. External maps are disjoint from the signed advantage head fit and its held physical maps but previously exposed to other development researches. Not new formal data or an independent statistical confirmation of win rate.')
    raw=(json.dumps(doc,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    p=HERE/'protocol.json'
    if p.exists():assert p.read_bytes()==raw
    else:p.write_bytes(raw)
    print(sha(raw),len(tickets),len(ref['steps']))
if __name__=='__main__':main()

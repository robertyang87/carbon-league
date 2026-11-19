"""Post hoc one-step study of the opening seen on the development maps. Never generate or run an examiner."""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sys
from shared import ACTOR,HERE,PROTO,ROOT,TRAIN,module,read,require,save,sha

CACHE=Path('/workspace/sapling-all-top5-20261119/replays')


def key(cmd):
    n=Counter(cmd.values())
    return f"{n['RECCOLLECTOR']}C{n['RECPLANTER']}P"


def run():
    plan=read(PROTO);panel=read(TRAIN/'panel.json')['games'];hist=[g for g in panel if g['split'] in ('val','holdout')]
    fit=[g for g in panel if g['split']=='train'];require(len(hist)==28 and all(not ({g[k] for g in hist}&{g[k] for g in fit}) for k in ('id','seed','terrain_sha256')),'physical alias in diagnostic')
    paths={'parent':ROOT/plan['arm_parent']['path'],'candidate':TRAIN/'serving/candidate.py','control':TRAIN/'serving/control.py'}
    mods={a:module(p,'diag_opening_'+a) for a,p in paths.items()}
    rows=[]
    for r in hist:
        p=CACHE/(r['id']+'.json.gz');require(sha(p)==r['sha256'],'source bytes')
        raw=read(p);s=r['seat'];q=raw['steps'][0];shared=q[0]['observation'];private=q[s]['observation']
        # Ladder encoding stores player lists and trees in seat 0's shared fields. Drop all spectator keys.
        obs={'player':s,'step':0,'players':shared['players'],'trees':shared['trees'],'carbon':private['carbon'],'remainingOverageTime':private['remainingOverageTime']}
        require(not obs['players'][s][2] and len(obs['players'][s][1])==2,'expected two empty bases')
        cmds={a:m.agent(obs,raw['configuration']) for a,m in mods.items()}
        logged=raw['steps'][1][s]['action'] or {}
        require(all(set(c)<=set(obs['players'][s][1]) and all(v in ('RECCOLLECTOR','RECPLANTER',None) for v in c.values()) for c in [logged,*cmds.values()]),'opening commands')
        rows.append(dict(id=r['id'],seed=r['seed'],terrain_sha256=r['terrain_sha256'],seat=s,opponent=r['opponent'],split=r['split'],wdl=r['wdl'],
                         logged_master6=logged,agent=cmds,composition={a:key(c) for a,c in {'recorded_teacher':logged,**cmds}.items()}))
    grouped={}
    for scope,part in [('all',rows)]+[(sp,[r for r in rows if r['split']==sp]) for sp in ('val','holdout')]:
        grouped[scope]={'maps':len(part),'composition':{a:dict(Counter(r['composition'][a] for r in part)) for a in ('recorded_teacher',*mods)},
                        'matched_full_teacher_command':{a:sum(r['agent'][a]==r['logged_master6'] for r in part) for a in mods}}
    doc={'category':'post-hoc after observing the first three native map/seat pairs opening', 'scope':'only the previously exposed 10 val and 18 holdout physical maps and true Master6 seat at turn 0; no agent call observes an already submitted current action',
         'does_not_identify_causal_terminal_gain':True,'registered_arena_unchanged':True,'panel_sha256':sha(TRAIN/'panel.json'),'packages':{a:sha(p) for a,p in paths.items()},'result':grouped,'rows':rows}
    save(HERE/'opening_transfer.json.gz',doc);print(grouped)

if __name__=='__main__':run()

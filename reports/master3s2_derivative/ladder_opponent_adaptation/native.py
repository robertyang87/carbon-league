"""Native calls on fixed archived loss roots with paired, repeated V2 and pupil."""
import argparse
import ast
import copy
import gzip
import hashlib
import json
import os
import time
import numpy as np
from register import HERE,BASE,sha
from study import P, model,frames,write
from build import assemble

# Use the existing local receipt implementation verbatim without importing older registry side effects.
root_helper=HERE.parent/'advantage_policy_update/probe.py'
source=root_helper.read_text()
node=next(f for f in ast.parse(source).body if isinstance(f,ast.FunctionDef) and f.name=='replay_one')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(root_helper),'exec'),globals())

def run():
    base=BASE.read_bytes();new=assemble();parents={'v2':model(base),'adapt':model(new)};m=parents['v2']
    assert all(p._S_ABORT==.58 and p._TEMP==1 and p._SEARCH_OK for p in parents.values())
    costs=list(m._S_COST)
    books=[]
    for id_ in P['held_ids']:
        card=min((r for r in P['cards'] if r['split']=='held' and r['opponent_id']==id_ and r['margin']<0),key=lambda r:r['match'])
        s=card['seat'];tr=m.BeliefTracker(m._Bel,s);hist=None
        for t,o,a,cfg in frames(card):
            if t in (44,108):
                seed=int(sha(f'v2-adapt/{id_}/{t}'.encode())[:8],16)
                sequence=['v2','adapt','adapt','v2'] if len(books)%2==0 else ['adapt','v2','v2','adapt']
                runs=[]
                for tag in sequence:
                    try:q=replay_one(parents[tag],tag,o,cfg,tr,hist,seed,costs)
                    except AssertionError:
                        # Public search can legitimately return the sole candidate before any enemy sample is needed.
                        assert not parents[tag]._S_STATS.get('last') and len(parents[tag]._S_REC)==0
                        q={'mode':tag,'single':True}
                    runs.append(q)
                row=dict(match=card['match'],terrain=card['terrain'],seat=s,t=t,opponent=card['opponent'],runs=runs)
                if all('issued' in v for v in runs):
                    keys=['issued','enemy','depths','K','H','M','menu']
                    a0,b0,b1,a1=runs
                    repeated=all(a0[k]==a1[k] and b0[k]==b1[k] for k in keys)
                    same_menu=a0['menu']==b0['menu']
                    same_work=all(a0[k]==b0[k] for k in ('K','H','M')) and sorted(a0['depths'])==sorted(b0['depths'])
                    changed=a0['issued']!=b0['issued'];diff=a0['enemy']!=b0['enemy']
                    row.update(repeated=repeated,same_menu=same_menu,identical_counts=same_work,same_depths=a0['depths']==b0['depths'],
                               enemy_changed=diff,issued_changed=changed,credited=changed and diff and same_menu and same_work and repeated,
                               same_index_credit=changed and diff and same_menu and a0['depths']==b0['depths'] and same_work and repeated)
                books.append(row)
            tr.step(o['players'],t,o['carbon']);tr.set_actions(o['players'],a[s]);hist=o['players']
            if t>=108:break
    counts={k:sum(r.get(k,False) for r in books) for k in ('repeated','same_menu','identical_counts','same_depths','enemy_changed','issued_changed','credited','same_index_credit')}
    raw=(json.dumps(dict(candidate=sha(new),parent=sha(base),protocol=sha((HERE/'protocol.json').read_bytes()),counts=counts,roots=books),ensure_ascii=False,allow_nan=False)+'\n').encode()
    (HERE/'native.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    print(counts,flush=True)
if __name__=='__main__':run()

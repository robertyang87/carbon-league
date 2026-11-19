"""Recreate real public roots, match full command dictionaries, and store the legal multi-option menu."""
from collections import defaultdict
from data import D,ROOT,PARENT,FastCarbon,load_parent,read,sha,compact
from actions import _jrc_context,_jrc_repr,_jrc_options
import gzip,hashlib,json
import numpy as np


def split(terrain,reg):
    return reg['split_by_terrain'].get(terrain,'held' if int(hashlib.sha256(('joint-rank-v1|'+terrain).encode()).hexdigest(),16)%3==0 else 'train')


def main():
    doc=read(D/'signed_roots.json.gz'); reg=read(D/'source_registry.json'); by=defaultdict(lambda:defaultdict(list))
    for row in doc['rows']:
        by[row['source']][row['t']].append(row)
    result=[];excluded=[]; feats=[];m=load_parent()
    for source,at in sorted(by.items()):
        path=ROOT/source;raw=read(path);s=raw['candidate_seat'];cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=raw['seed'])
        for t,commands in enumerate(raw['actions']):
            if t>max(at):break
            obs=sim.obs(s)
            if t==0 or t in at:
                root=m.agent(obs,cfg);assert root==commands[s],(source,t)
            else:m._FEAT.observe(obs)
            if t in at:
                x,aux,emb,y=m._jrc_root;ctx=_jrc_context(obs,cfg,m._FEAT,x,aux,emb)
                options=_jrc_options(ctx,obs,cfg,m._FEAT,x,aux,y)
                menu=[m._FEAT.to_actions(o[0]) for o in options];loc=len(feats);feats.extend(o[1] for o in options)
                for r in at[t]:
                    assert r['source_sha256']==sha(path) and r['terrain']==raw['initial_terrain_u8_sha256'] and r['parent']==root
                    alt=r['alternative']; lab=m._FEAT.labels(alt)
                    if m._FEAT.to_actions(lab)!=alt:
                        excluded.append(dict(case=r['case'],reason='unsupported or unrepresented full command'));continue
                    valid=_jrc_options(ctx,obs,cfg,m._FEAT,x,aux,y,[(lab,r['role'])])
                    if len(valid)!=2:
                        excluded.append(dict(case=r['case'],reason='introduces illegal prefix or new same-team conflict/cash insufficiency'));continue
                    if alt in menu:
                        idx=loc+menu.index(alt); in_menu=True
                    else:
                        idx=len(feats);feats.append(valid[1][1]); in_menu=False
                    # Entire joint action matches. Never match just a worker or previously logged slot.
                    result.append(dict(case=r['case'],split=split(r['terrain'],reg),terrain=r['terrain'],seed=r['seed'],t=t,seat=s,role=r['role'],family=r['family'],
                         parent=loc,alternate=idx,menu=[loc,loc+len(menu)],in_menu=in_menu,menu_commands=menu,menu_roles=[o[2] for o in options],
                         signed_parent=root,signed_alternative=alt,delta=[r['alternative_scores'][i]-r['parent_scores'][i] for i in (s,1-s)]))
            sim.step(*commands)
    np.savez_compressed(D/'native_inputs.npz',features=np.stack(feats))
    out=dict(source_sha256=sha(D/'signed_roots.json.gz'),rows=result,excluded=excluded)
    (D/'native_inputs.json.gz').write_bytes(gzip.compress(compact(out),mtime=0))
    print('native input',len(result),len(excluded),len(feats),flush=True)


if __name__=='__main__':main()

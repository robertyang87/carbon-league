"""Separate package identity, actor tensors, historical paired outcomes and alternate opponent evidence."""
import ast
import base64
from collections import Counter
import hashlib
import json
import math
import statistics
import subprocess
from build import HERE,ROOT,FROZEN,EXPERT,compact,payload,digest,out

HIST='d599ec679718dd4407a718aa6ad810e35e2d3983'
ASSETS={
   'it50':'code/packs/ppou2_m_it50.py','it45':'code/packs/ppou2_m_it45.py','it40_echo5':'code/packs/ppou2_m_it40.py',
   'echo4':'code/packs/echo4_int8.py','echo6':'sources/echo6/echo4_search_int8_d1_homefix.py',
   'master3s2':'sources/master3s2/master3s2.py','echo7':'code/apexcarbon_v4tta_search.py',
   'apex_fix1':'code/apexcarbon_v1_fixed.py','apex_fix2':'code/apexcarbon_v1_fix2.py','apex_v2':'code/apexcarbon_v2.py'}
MATCHES={'it50':'abD_it50_3s2','echo4':'f2_echo4_3s2','echo6':'f1_comp_3s2'}


def blob(path,ref=FROZEN):
    return subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=ROOT)


def stats(rows):
    m=[x['margin'] for x in rows];n=len(rows)
    return dict(n=n,wins=sum(y>0 for y in m),draws=sum(y==0 for y in m),mean_margin=statistics.mean(m))


def main():
    assets={};corpora={};rows={}
    for name,path in ASSETS.items():
        raw=blob(path)
        a=assets[name]=dict(path=path,pack_sha256=digest(raw),pack_md5=hashlib.md5(raw).hexdigest(),bytes=len(raw))
        if name.startswith('apex') or name in ('master3s2','echo7'):
            weights=payload(raw,('_WEIGHTS','_VALUE'))
            a.update(policy_npz_sha256=digest(base64.b64decode(weights['_WEIGHTS'])),critic_npz_sha256=digest(base64.b64decode(weights['_VALUE'])))
        else:
            weights=payload(raw,EXPERT);a['stored_actor_with_metadata_sha256']=digest(compact(weights));a['actor_meta']=weights['_M1']
    for k,v in assets.items():
        if 'policy_npz_sha256' in v:
            v['both_npz_identical_to_original']=all(v[n]==assets['master3s2'][n] for n in ('policy_npz_sha256','critic_npz_sha256'))
    for name,tag in MATCHES.items():
        path='synced/h29/bigeval_live/'+tag+'.jsonl'
        raw=blob(path,HIST);lbl=[json.loads(x) for x in raw.splitlines()]
        require=(len(lbl)==len({x['seed'] for x in lbl}) and all(x['ah']==assets[name]['pack_md5'] and
                 x['bh']=='8d209ddb46faafae151aae16b7e17d55' and x['hh']=='354bcd1bc2dc8bc188c813f53d28c893' and
                 x['err'] is None and x['serr']==x['berr']==0 and x['side']==x['seed']%2 and abs(x['margin']-x['sa']+x['sb'])<1e-7 for x in lbl))
        if not require:raise ValueError('historical series identity')
        rows[name]={x['seed']:x for x in lbl}
        corpora[name]=dict(path=path,sha256=digest(raw),all_games=stats(lbl),opponent_stats=sorted({a for x in lbl for a in x['bst']}))
    shared=sorted(set.intersection(*(set(d) for d in rows.values())))
    if len(shared)!=300:raise ValueError('expected precisely all it50 maps')
    paired={}
    for name in ('echo4','echo6'):
        group=[rows[name][s] for s in shared]
        md=[x['margin']-rows['it50'][x['seed']]['margin'] for x in group]
        wd=[int(x['margin']>0)-int(rows['it50'][x['seed']]['margin']>0) for x in group]
        half=statistics.stdev(md)/math.sqrt(len(md))*1.96
        paired[name]=dict(**stats(group),by_physical_seat=[stats([x for x in group if x['side']==k]) for k in (0,1)],
                          avg_delta_margin=statistics.mean(md),normal95_delta_margin=[statistics.mean(md)-half,statistics.mean(md)+half],
                          wins_only_this=sum(v==1 for v in wd),wins_only_it50=sum(v==-1 for v in wd),wins_delta=statistics.mean(wd))
    tree=subprocess.check_output(['git','ls-tree','-lr',FROZEN],cwd=ROOT,text=True).splitlines()
    size_hits=[];name_hits=[]
    for line in tree:
        desc,path=line.split('\t',1)
        if desc.split()[-1]=='5747509':size_hits.append(path)
        if any(x in path.lower() for x in ('ppou2_m_it55','ckpt_55','m55_e4','it55_echo4')):name_hits.append(path)
    out(HERE/'evidence.json',dict(frozen_main=FROZEN,archive=HIST,assets=assets,
        original_registered_historical=corpora,on_shared_300=paired,historical_original_search_per_turn='not recorded',
        it55_known_md5='b42a6c3438530d9cfebe27abb96d81c6',it55_exact_5747509byte_tree_paths=size_hits,
        it55_known_name_tree_paths=name_hits))


if __name__=='__main__':
    main()

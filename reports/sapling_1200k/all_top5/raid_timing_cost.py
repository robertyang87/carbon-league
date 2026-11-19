"""Explicit one-step deposit opportunity costs and transfer sensitivity by opponent."""
import argparse
import gzip
import json
import re
from pathlib import Path
from hashlib import sha256
import numpy as np
from raid_timing import HERE,DEST,DIST,JOINT,intervals,sample_stats
from defense_run import atomic,compact


def bases(path):
    with gzip.open(path,'rb') as f:
        data=b''
        while len(data)<2**20:
            chunk=f.read(32768)
            if not chunk:break
            data+=chunk
            s=data.decode('utf8',errors='ignore');hit=re.search(r'"steps"\s*:\s*\[',s)
            if not hit:continue
            try:frame=json.JSONDecoder().raw_decode(s[hit.end():].lstrip())[0]
            except ValueError:continue
            cfg_start=re.search(r'"configuration"\s*:\s*',s)
            cfg=json.JSONDecoder().raw_decode(s[cfg_start.end():].lstrip())[0]
            return [tuple(p[1].values()) for p in frame[0]['observation']['players']],int(cfg['episodeSteps'])-1
    raise ValueError('unreadable first frame')


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--cache',required=True,type=Path);args=a.parse_args()
    d=json.loads(gzip.decompress((HERE/'raid_timing.json.gz').read_bytes()))
    docs=d['games'];x=np.load(HERE/'raid_timing_rows.npz')['rows'];rs=x['non_enemy_base_cells']>0
    board=[bases(args.cache/'replays'/(g['id']+'.json.gz')) for g in docs]
    bs,deadlines=zip(*board);deadline=np.asarray(deadlines)[x['game']]
    choices=np.where(rs&(x['could_bank_now']>0))[0]
    both=np.zeros(len(x),bool);away=np.zeros(len(x),bool)
    for i in choices:
        g,s,wp,vp,move=(int(x[name][i]) for name in ('game','seat','hunter_cell','victim_cell','our_action'))
        own,opp=bs[g][s],bs[g][1-s];cross=[p for p in JOINT[wp][vp] if p not in opp]
        if not any(p in own for p in DEST[wp]):raise ValueError('cash option feature out of sync')
        both[i]=any(q in own for q in cross);away[i]=DEST[wp][move] not in own and x['selected_crossing'][i]
    rnk=np.asarray([g['top5'] for g in docs],dtype=object);rnk=np.asarray([[n or 0 for n in row] for row in rnk],dtype=int)
    opp=rnk[x['game'],1-x['seat']]
    group={'own_bank_available':rs&(x['could_bank_now']!=0),'own_bank_and_at_least_one_bank_crossing':both,
           'own_bank_but_all_crossings_require_leaving_delivery':rs&(x['could_bank_now']!=0)&~both,
           'actually_selected_crossing_away_from_own_direct_delivery':away}
    # Only compare exit counts when the same collector pair was in range on the prior frame.
    step=x['step'].astype('i8')
    unit_limit=1+max(int(x['hunter'].max()),int(x['victim'].max()))
    pair=((x['game'].astype('i8')*2+x['seat'])*unit_limit+x['hunter'])*unit_limit+x['victim']
    stamp=pair*1024+step;order=np.argsort(stamp);keys=stamp[order]
    j=np.clip(np.searchsorted(keys,stamp-1),0,len(keys)-1)
    seen_prev=(step>0)&(keys[j]==stamp-1)
    prev_exits=np.where(seen_prev,x['safe_exits'][order[j]],-1)
    for label,mask in [('no_prior_pair',prev_exits<0),('already_boxed',prev_exits==0),
                       ('was_one',prev_exits==1),('was_two_plus',prev_exits>=2)]:
        group['zero_exits_'+label]=rs&(x['safe_exits']==0)&mask
    # Earliest home arrival for any eligible intersection. Root t -> collision t+1, then walk home.
    # The game can end by extinction earlier; use the scheduled step limit, knowable live.
    room=np.full(len(x),-2048,dtype='i2')
    layouts={}
    for i,layout in enumerate(bs):layouts.setdefault(tuple(layout),[]).append(i)
    for layout,indices in layouts.items():
        layout_mask=np.isin(x['game'],indices)
        for seat,(own,enemy) in enumerate((layout,layout[::-1])):
            hops=np.full((225,225),255,dtype='i2')
            for w in range(225):
                for v in range(225):
                    cross=[q for q in JOINT[w][v] if q not in enemy]
                    if cross:hops[w,v]=min(DIST[q][h] for q in cross for h in own)
            m=rs&layout_mask&(x['seat']==seat)
            room[m]=deadline[m]-1-step[m]-hops[x['hunter_cell'][m],x['victim_cell'][m]]
    for k,m in [('impossible',room<0),('exact',room==0),('one_two', (room>=1)&(room<=2)),('three_plus',room>=3)]:
        group['scheduled_bank_slack_'+k]=rs&m
    for r in range(6):
        for name,rem in [('boxed',x['safe_exits']==0),('one',x['safe_exits']==1),('free',x['safe_exits']>=2)]:
            group[f'opponent_{r}_{name}']=rs&(opp==r)&rem
    unique,counts=np.unique([g['seed'] for g in docs],return_inverse=True);boot=np.random.default_rng(93921).integers(0,len(unique),(600,len(unique)))
    ci=intervals(x,counts,boot,list(group.values()))
    result={k:dict(**sample_stats(x,m),own_pre_cargo_exposure=round(float(x['our_cargo'][m].sum(dtype='f8')),3),cluster_ci=u) for (k,m),u in zip(group.items(),ci)}
    atomic(HERE/'raid_timing_cost.json',compact({'scheduled_terminal_frames':sorted(set(deadlines)),'base_layout_frequencies':{str(v):sum(str(b)==v for b in bs) for v in set(map(str,bs))},'groups':result}))

if __name__=='__main__':main()

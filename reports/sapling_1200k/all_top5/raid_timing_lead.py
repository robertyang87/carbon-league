"""Use current observable dwell time and future event windows without feeding future labels as predictors."""
import argparse
import gzip
import io
import json
from pathlib import Path
import numpy as np
from raid_timing import HERE,intervals
from defense_run import atomic,compact
from hashlib import sha256

def main():
    p=np.load(HERE/'raid_timing_rows.npz',allow_pickle=False);x=p['rows']
    doc=json.loads(gzip.decompress((HERE/'raid_timing.json.gz').read_bytes()));games=doc['games']
    ids=np.asarray(x['game'],dtype='i8');seats=x['seat'];t=x['step']
    unit_count=max(int(x['hunter'].max()),int(x['victim'].max()))+1
    pair=((ids*2+seats)*unit_count+x['hunter'])*unit_count+x['victim']
    time=x['step'].astype('i8');stamp=pair*1024+time
    sort=np.argsort(stamp);keys=stamp[sort]
    assert np.all(np.diff(keys)>0)
    row=np.arange(len(keys));start=np.maximum.accumulate(np.where(np.r_[True,np.diff(keys)!=1],row,0))
    dwell=np.empty_like(t);dwell[sort]=row-start+1
    useful=x['non_enemy_base_cells']>0;nsteps=np.asarray([g['frames'] for g in games])[ids]
    head={};data={'visible_consecutive_overlap_turns':dwell.astype('i2')}
    for lab,col in [('capture','captured'),('cash','banked')]:
        positive=stamp[x[col]>0]
        assert len(positive)>0
        order=np.sort(positive)
        nxt=np.searchsorted(order,stamp)
        near=order[np.minimum(len(order)-1,nxt)]
        time_to=np.where((nxt<len(order))&(near//1024==pair),near-stamp,999)
        for h in (1,3,5,10):
            elapsed=nsteps-t>=h
            y=(time_to<h);eligible=useful&elapsed
            data[f'observed_{lab}_within_{h}']=np.where(elapsed,y,-1).astype('i1')
            groups={'any':eligible,'0 exits':eligible&(x['safe_exits']==0),'1 exit':eligible&(x['safe_exits']==1),
                     '2+ exits':eligible&(x['safe_exits']>=2),'overlap first turn':eligible&(dwell==1),
                     'overlap second turn':eligible&(dwell==2),'overlap 3-5 turns':eligible&(dwell>=3)&(dwell<=5),
                     'overlap 6+ turns':eligible&(dwell>=6)}
            head[f'{lab}_{h}']={k:{'eligible':int(m.sum()),'observed_positive':int(y[m].sum())} for k,m in groups.items()}
    # For realized banked hits, identify whether this very pair was already in the geometric risk set on earlier turns.
    trail={}
    for label,positive in [('captured',x['captured']>0),('banked',x['banked']>0),('missed',x['missed_conditional']>0)]:
        bins={}
        for d in (1,3,5,10):
            # Stamps are decision-state times. Contact frame T is that decision's t+1.
            offset=d-1
            eligible=positive&(t>=offset)
            prev=stamp[eligible]-offset;idx=np.searchsorted(keys,prev);idx=np.minimum(idx,len(keys)-1)
            found=keys[idx]==prev;valid=sort[idx[found]]
            bins[str(d)]={'outcomes':int(eligible.sum()),'same_pair_was_threat':int(found.sum()),
                          'same_pair_then_zero_exits':int((x['safe_exits'][valid]==0).sum())}
        trail[label]=bins
    out=io.BytesIO();np.savez_compressed(out,**data);atomic(HERE/'raid_timing_lead.npz',out.getvalue())
    atomic(HERE/'raid_timing_lead.json',compact({'source_rows_sha256':doc['rows_sha256'],'feature':'visible_consecutive_overlap_turns',
          'outcomes':'Same pursuer and original carrier. Includes the current transition. At a near-terminal root with fewer than h transitions, label is -1.',
          'lead_rows_sha256':sha256(out.getvalue()).hexdigest(),'prospective':head,'retrospective':trail}))
    print(head['cash_5'],trail['banked'])

if __name__=='__main__':main()

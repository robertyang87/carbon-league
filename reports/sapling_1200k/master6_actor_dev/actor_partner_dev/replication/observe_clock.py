"""Read existing native progress without changing a worker's clock, affinity or search."""
from collections import Counter
import argparse
import gzip
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
def main(owner):
    last={};observations={};external_times={}
    out=HERE/'live_clock.json.gz'
    if out.exists():return
    while not (HERE/'comparison.json').exists():
        stat=Path(f'/proc/{owner}/stat')
        if not stat.exists() or stat.read_text().split(') ')[1].split()[0]=='Z':
            break
        for p in (HERE/'games').glob('*/progress_*.json.gz'):
            key=p.parent.name+'/'+p.name;prior=last.get(key,{})
            try:
                stamp=p.stat().st_mtime_ns
                if prior.get('stamp')==stamp:continue
                d=json.loads(gzip.decompress(p.read_bytes()))
            except FileNotFoundError:continue
            step=d['step'];hist=Counter({int(k):v for k,v in d['stats']['depth'].items()})
            before=prior.get('hist',Counter());pre_t=prior.get('t',0)
            delta=hist-before
            rec=dict(previous_transition=pre_t,transition=step,delta_depth_hist=dict(delta),
                     absolute_depth_hist=dict(hist),last_return_label=d['search'][-1][1],cum_searched=d['stats']['searched'],
                     cum_aborted=d['stats']['aborted'],warm_cost=d['cost'],root_cpu_time_observation='wall time in source stats')
            if step==pre_t+1:
                if sum(delta.values())==1:rec['exact_depth_if_observed']=next(iter(delta))
                elif not delta:rec['exact_depth_if_observed']=None
            observations.setdefault(key,[]).append(rec);last[key]={'t':step,'hist':hist,'stamp':stamp}
            external_times[key]=d['times']
        time.sleep(.12)
    data=dict(source='read-only existing atomic search checkpoints; collection started mid-study, exact depths only when an adjacent pair of checkpoints proves them',
              external_call_seconds_by_physical_seat=external_times,per_arm=observations,first_sample_unobserved_earlier_steps=True)
    temp=out.with_suffix('.tmp');temp.write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0));os.replace(temp,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('owner',type=int);main(parser.parse_args().owner)

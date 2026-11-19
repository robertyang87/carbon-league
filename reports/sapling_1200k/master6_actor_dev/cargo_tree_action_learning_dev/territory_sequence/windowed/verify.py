"""Keep all signed historical observations, public replies and latencies for the newly frozen package."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
EXP=STUDY.parent
OLD=EXP/'ladder_timing'
TAPES=STUDY/'ladder_timing'
spec=importlib.util.spec_from_file_location('paired_timing',TAPES/'run.py')
library=importlib.util.module_from_spec(spec);spec.loader.exec_module(library)
bench=library.bench

def load(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())

def main():
    pack=load(HERE/'pack.json');prot=load(HERE/'protocol.json')
    assert bench.digest(HERE/'candidate.py')==pack['candidate'] and bench.digest(HERE/'protocol.json')==pack['protocol']
    assert os.sched_getaffinity(0)=={2}
    pred=load(TAPES/'summary.json');op=load(OLD/'protocol.json')
    assert bench.digest(OLD/'replay.py')==op['client'] and bench.digest(OLD/'worker.py')==op['worker']
    assert pred['candidate']==prot['source']['all_steps'] and pred['parent']==prot['source']['parent']
    bench.HERE=HERE/'timing';steps=[];receive0=bench.receive
    def receive(proc,timeout):
        msg=receive0(proc,timeout)
        if 'step' in msg:steps.append(msg['action'])
        return msg
    bench.receive=receive
    rows=[];walls=[];seats=set();n_changed_full=0
    for fn in sorted(op['raw']):
        stamps=fn.replace('raw_','timing_');pc=TAPES/('parent_'+stamps);hc=TAPES/('candidate_'+stamps)
        old={arm:load(path) for arm,path in [('parent',pc),('h24',hc)]}
        assert all(any(bench.digest(path)==v['sha256'] for v in pred['receipts'][arm]) for arm,path in [('parent',pc),('candidate',hc)])
        out=HERE/'timing'/stamps
        if out.exists():
            rec=load(out)
        else:
            steps.clear()
            rec=bench.collect(EXP/'direct_original/games/candidate'/fn,dict(op,candidate=pack['candidate']))
            assert len(steps)==rec['n']==299
            rec.update(public_actions=list(steps),source=op['raw'][fn],parent_tape=bench.digest(pc),h24_tape=bench.digest(hc))
            library.write(out,rec)
        assert (rec['source'],rec['parent_tape'],rec['h24_tape'])==(op['raw'][fn],bench.digest(pc),bench.digest(hc))
        assert rec['ready']['sha256']==pack['candidate'] and rec['ready']['numpy']=='1.26.4' and rec['ready']['python'].startswith('3.12.')
        assert rec['ready']['affinity']==[2] and rec['ready']['rlimit_as']==[536870912,536870912]
        assert rec['ready']['cgroup']==old['parent']['ready']['cgroup']==old['h24']['ready']['cgroup']
        assert rec['final_cash']==old['parent']['final_cash']==old['h24']['final_cash']
        li,hi=prot['window'];wrong_in=[];wrong_out=[];effective=[];diff_full=[];u=set()
        for t,(actual,past,whole,step) in enumerate(zip(rec['public_actions'],old['parent']['public_actions'],old['h24']['public_actions'],rec['steps'],strict=True)):
            assert step['t']==t
            if li<=t<=hi:
                if actual!=whole:wrong_in.append(t)
                if actual!=past:
                    effective.append(t)
                    u.update((t,k) for k in set(actual)|set(past) if actual.get(k)!=past.get(k))
            else:
                if actual!=past:wrong_out.append(t)
                if whole!=past:diff_full.append(t)
        if effective:seats.add(rec['seat'])
        n_changed_full+=len(diff_full)
        s=rec['steps'];sec=[r['external_s'] for r in s];walls+=sec
        assert len(sec)==299
        row=dict(seed=rec['seed'],seat=rec['seat'],n=len(sec),in_window=hi-li+1,wrong_in=wrong_in,wrong_out=wrong_out,
            changed_vs_parent_in=effective,changed_physical_commands_in=len(u),past_h24_disagreements_outside=len(diff_full),
            step0_ms=sec[0]*1e3,step128_ms=sec[128]*1e3,step160_ms=sec[160]*1e3,init_s=rec['init_external_s'],
            p95_ms=rec['p95_s']*1e3,p99_ms=rec['p99_s']*1e3,max_ms=max(sec)*1e3,slow=rec['slow_steps'],excess_s=rec['sum_excess_s'],
            peak_rss_mib=rec['peak_rss_kib']/1024,wall_s=rec['wall_start_to_process_exit_s'],local_thresholds=rec['local_thresholds_pass'],receipt=bench.digest(out))
        rows.append(row)
        print(row['seed'],row['seat'],len(effective),len(wrong_in),len(wrong_out),row['slow'],flush=True)
    eligible=all(not r['wrong_in'] and not r['wrong_out'] and r['local_thresholds'] for r in rows) and seats=={0,1} and n_changed_full>0
    results=dict(sha=pack['candidate'],bytes=pack['bytes'],generated_utc=datetime.now(timezone.utc).isoformat(),total_calls=len(walls),
                 local_precheck_passed=eligible,inside=sum(r['in_window'] for r in rows),outside=sum(r['n']-r['in_window'] for r in rows),
                 effective_inside=sum(len(r['changed_vs_parent_in']) for r in rows),effective_outside_removed=n_changed_full,
                 p95_ms=float(np.percentile(walls,95)*1000),p99_ms=float(np.percentile(walls,99)*1000),
                 max_ms=float(max(walls)*1000),slow=sum(r['slow'] for r in rows),excess_s=sum(r['excess_s'] for r in rows),
                 cgroup=rec['ready']['cgroup'],rows=rows,formal_ladder_certification=False)
    library.write(HERE/'qualification.json',results)
    assert eligible

if __name__=='__main__':main()

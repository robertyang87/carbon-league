"""Paired ABBA CPU timings on identical archived whole-search neural workloads."""
import itertools
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from capture_spatial import module, original, sha, HERE
from network import predict, OBSERVED, Readout


def main():
    torch.set_num_threads(1);net=module(original())._net();model=Readout()
    with np.load(HERE/'student.npz') as f:
        model.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files})
    panel=OBSERVED/'timing';manifest=json.loads((panel/'panel.json').read_text());rows=[];errors={}
    assert sha((panel/'panel.npz').read_bytes())==manifest['features_sha256']
    with np.load(panel/'panel.npz') as raw:
        for ix,card in enumerate(manifest['public_roots']):
            xs=[]
            for b in card['batches']:
                x=raw[b['key']].astype(np.float32)*np.float32(.01)
                sc=raw[b['key']+'s'] if b['scalar'] else None
                xs.append((b['kind'],x,b['sid'],sc))
            def run(mode):
                for kind,x,sid,sc in xs:
                    if mode=='original' or kind not in ('joint','other'):
                        w,b,v=net(x,sid,sc)
                        if kind=='joint':
                            tmp=w[::2],b[::2],v[::2],w[1::2],b[1::2]
                        else:tmp=w,b,v
                    elif kind=='joint':
                        w,b,v=net(np.ascontiguousarray(x[::2]),sid,np.ascontiguousarray(sc[::2]))
                        ow,ob=predict(net,model,np.ascontiguousarray(x[1::2]),sid)
                        tmp=w,b,v,ow,ob
                    else:tmp=predict(net,model,x,sid)
            for kind,x,sid,sc in xs:
                if kind!='joint':continue
                w,b,v=net(x,sid,sc);w2,b2,v2=net(np.ascontiguousarray(x[::2]),sid,np.ascontiguousarray(sc[::2]))
                for k,a,c in [('own_worker',w[::2],w2),('own_base',b[::2],b2),('own_value',v[::2],v2)]:
                    e=float(abs(a-c).max());errors[k]=max(errors.get(k,0.),e)
            run('original');run('student')
            times={k:[] for k in ('original','student')}
            sequence=['original','student','student','original']
            if ix%2:sequence=sequence[::-1];sequence=['student' if k=='original' else 'original' for k in sequence]
            for mode in sequence:
                t=time.perf_counter();c=time.process_time();run(mode)
                times[mode].append([time.perf_counter()-t,time.process_time()-c])
            rows.append(dict(seed=card['seed'],seat=card['seat'],t=card['t'],batches=len(xs),
                        neural_rows=sum(len(x) for _,x,_,_ in xs),times=times,
                        medians={m:[statistics.median(v[i] for v in d) for i in range(2)] for m,d in times.items()}))
    out=dict(source=sha(original()),features_sha=manifest['features_sha256'],weights_sha=sha((HERE/'student.npz').read_bytes()),
             measured='ABBA/BAAB per fixed root, median two samples; complete fixed neural workload including extracting actor slots, gathering output and all unchanged own neural/critic calls; no simulation, endpoint, init or resource certification',affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),
             own_output_max_abs=errors,cases=rows,sums={m:[sum(r['medians'][m][i] for r in rows) for i in range(2)] for m in ('original','student')})
    (HERE/'timing.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out['sums'],flush=True)


if __name__=='__main__':main()

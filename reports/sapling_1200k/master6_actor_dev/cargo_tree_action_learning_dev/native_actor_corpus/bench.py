import time
import torch
from common import *
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
m=package(); c=read(EXP/'panel.json')['cards'][0]
f=sys.modules['cnet2_feat'].Feat(1-c['seat'])
rows=[]; origin=time.monotonic();timings=[]
for t,obs,cfg,act,selected in iter_public(c,6):
    x=f.observe(obs)
    if selected:
        t0=time.monotonic()
        y=m._mixed_actions(x,f.aux)
        elapsed=time.monotonic()-t0
        r=actual_features(f,x,act,m)
        rows.append(r);timings.append(elapsed)
bench_hist=time.monotonic()-origin
d={k:np.stack([r[k] for r in rows]) for k in rows[0]}
b=tensorbatch(d,np.arange(len(rows)));student=torch_model(m,1);frozen=torch_model(m,0)
times=[]
for _ in range(3):
    start=time.monotonic()
    with torch.no_grad():w=frozen(b)[0]
    pred=(.75*w+.25*student(b)[0]+b['residual']).masked_fill(~b['legal'],-1e9)
    loss=-pred.log_softmax(-1).gather(-1,b['y'].clamp_min(0)[...,None]).squeeze(-1)[b['eligible']].mean()
    loss.backward();student.zero_grad(set_to_none=True)
    times.append(time.monotonic()-start)
result=dict(threads=torch.get_num_threads(),affinity=sorted(os.sched_getaffinity(0)),torch=torch.__version__,numpy=np.__version__,
 cpu_threads=torch.get_num_threads(),raw=c['sha256'],public_history_steps=len(read(ROOT/c['path'])['actions']),history_and_six_decodes_s=bench_hist,
 six_forward_backward_s=times,raw_free_parent_forward_s=timings,loss=float(loss.detach()))
save(HERE/'benchmark.json',result);print(result)

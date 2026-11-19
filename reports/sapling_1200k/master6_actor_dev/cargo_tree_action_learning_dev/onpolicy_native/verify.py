"""Check the rounded deployment tensors and same-command probabilities against signed training roots."""
import numpy as np
from experiment import FIT,HERE,PARENT,ROOT,FastCarbon,model,read,sha,write
from build import build

def run():
    f=read(HERE/'fit.json')
    if 'selected' not in f: return
    target=HERE/'candidate.py'; new=model(target); old=model(PARENT)
    seed=621758345; seat=0
    raw=read(HERE/'games/candidate'/f'raw_{seed}_{seat}.json.gz')
    world=FastCarbon(raw['game_configuration']).reset(seed=seed)
    obs=world.obs(seat)
    for net in (old,new):net.agent(obs,world.config)
    # Full byte identity of both policy packages outside the student's quantized tensors is checked in fit.py.
    by_network=[]
    for no,(left,right) in enumerate(zip(old._NETS,new._NETS)):
        if set(left.w)!=set(right.w):raise ValueError('tensor set changed')
        changes={}
        for key in left.w:
            a,b=left.w[key],right.w[key]
            if not np.array_equal(a,b):changes[key]=int(np.count_nonzero(a!=b))
        if no==0 and changes:raise ValueError('frozen it50 network differs')
        if no==1 and not set(changes)<=set(('whead.weight','whead.weightT','whead.bias')):raise ValueError('other Sapling inference tensor changed')
        by_network.append(changes)
    dw=new._NETS[1].w['whead.weightT']-old._NETS[1].w['whead.weightT']
    db=new._NETS[1].w['whead.bias']-old._NETS[1].w['whead.bias']
    # Independent bounded-error check for the int8 form of the single registered step.
    with np.load(HERE/'candidate.npz',allow_pickle=False) as a,np.load(FIT/'candidate.npz',allow_pickle=False) as b:
        truew=(a['whead.weight']-b['whead.weight']).T
        trueb=a['whead.bias']-b['whead.bias']
        if any(not np.array_equal(a[k],b[k]) for k in a.files if k not in ('whead.weight','whead.bias')):
            raise ValueError('additional FP32 parameter drift')
    agg=[];fil=[]
    for g in read(HERE/'scores.json'):
        imp=[]
        for event in g['events']:
            if not event['supported']:continue
            p=HERE/'traces'/f"{g['seed']}_{g['seat']}"/f"{event['t']:03d}.json"
            if sha(p)!=event['sha256']:raise ValueError('unbound logged feature')
            e=read(p); h=np.asarray(e['hidden'],dtype=np.float64); acts=e['actions']; probs=np.asarray(e['probabilities'])
            training_delta=.25*(h@truew[:,acts]+trueb[acts])
            deployed_delta=.25*(h@dw[:,acts]+db[acts])
            def normalize(z):
                tmp=np.exp(z-z.max());return tmp/tmp.sum()
            q=normalize(np.array(e['logits'])+deployed_delta)
            diff=float(np.dot(probs,np.log(probs/q)))
            val=float(np.log(q[e['choice']]/probs[e['choice']]))
            fil.append(dict(seed=g['seed'],seat=g['seat'],t=e['step'],conditional_kl=diff,
                training_logit_gap_change=float(np.diff(training_delta)[0]),
                quantized_logit_gap_change=float(np.diff(deployed_delta)[0]),logp_change_chosen=val))
            imp.append(val)
        row=next(v for v in f['episodes'] if (v['seed'],v['seat'])==(g['seed'],g['seat']))
        agg.append(row['advantage']*sum(imp))
    if not np.isfinite(agg).all():raise ValueError('nonfinite projected gradient')
    write(HERE/'deployed_tensor_audit.json',dict(updated=sha(target),starting=sha(PARENT),fp32=sha(HERE/'candidate.npz'),
        observed_tensor_element_changes=by_network,max_paired_logit_quantization_error=max(abs(q['training_logit_gap_change']-q['quantized_logit_gap_change']) for q in fil),
        mean_conditional_kl=float(np.mean([r['conditional_kl'] for r in fil])),
        training_fixed_menu_advantage_times_logp_gain=float(np.mean(agg)),per_decision=fil,
        scope='Diagnostic on training data only, frozen parent prefixes/guard and feasible pair. New greedy deployment normally recomputes all prefixes and does not sample this distribution. Not an off-policy strength estimate.'))
    print(by_network,flush=True)
if __name__=='__main__':run()

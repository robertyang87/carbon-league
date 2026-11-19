"""One bounded on-policy Monte Carlo update of the exact existing Sapling worker readout."""
import math
import sys
import numpy as np
from experiment import ACTOR, FIT, HERE, PARENT, ROOT, SEEDS, read, replay, sha, write

def utility(scores, s):
    d=float(scores[s]-scores[1-s])
    wdl=1. if d>0 else .5 if d==0 else 0.
    return .5*wdl+.5/(1.+math.exp(-float(np.clip(d/1000.,-50.,50.))))

def direction(rows):
    grads=[r['gradient'] for r in rows]
    return np.mean(grads,0)/(np.mean([r['fisher'] for r in rows],0)+.025)

def run():
    p=read(HERE/'protocol.json')
    if p['procedure']['fit.py']!=sha(HERE/'fit.py'):raise ValueError('unregistered estimator')
    games=read(HERE/'scores.json')
    controls={(r['seed'],r['seat']):r for r in p['historical_controls']}
    if len(games)!=8 or {(r['seed'],r['seat']) for r in games}!=set(controls):raise ValueError('incomplete eight-game training batch')
    episodes=[]; evidence=[];diffs=[]
    choices={0:[],1:[]}
    for g in games:
        seed,seat=g['seed'],g['seat']
        rt=utility(g['score'],seat);bt=utility(controls[seed,seat]['scores'],seat)
        grad=np.zeros((5,97));fisher=np.zeros_like(grad);n=0
        for entry in g['events']:
            path=HERE/'traces'/f'{seed}_{seat}'/f"{entry['t']:03d}.json"
            if sha(path)!=entry['sha256']:raise ValueError('changed signed behavior trace')
            e=read(path)
            if not e['supported']:continue
            h=np.r_[np.asarray(e['hidden'],np.float64),1.]
            q=np.asarray(e['probabilities'])
            z=np.asarray(e['logits'],dtype=np.float64)
            opts=e['actions'];choice=e['choice']; checks=np.exp(z-z.max());checks/=checks.sum()
            if (h.shape!=(97,) or not np.isfinite(h).all() or not np.allclose(checks,q,rtol=0,atol=1.e-12)
                or len(set(opts))!=2 or not all(e['legal'][v] for v in opts)
                or abs(e['logp']-math.log(q[choice]))>1.e-12 or int(e['uniform']>=q[0])!=choice or
                e['issued']!=e['options'][choice]):
                raise ValueError('unsupported or inexact behavior likelihood')
            # Joint command options map one-to-one to the two possible focal-slot actions.
            # Features and both complete commands precede the private RNG draw.
            d=np.zeros((5,97));d[opts[0]]=.25*h;d[opts[1]]=-.25*h
            grad+=d*((choice==0)-q[0])*(rt-bt)
            fisher+=q[0]*q[1]*d*d
            choices[seat].append(choice);diffs.append(d);n+=1
        episodes.append(dict(seed=seed,seat=seat,gradient=grad,fisher=fisher))
        evidence.append(dict(seed=seed,seat=seat,sampled=n,score=g['score'],control=controls[seed,seat]['scores'],
                             official_U=rt,baseline_U=bt,advantage=rt-bt))
    total=len(diffs)
    if total<24 or any(set(x)!={0,1} for x in choices.values()):
        write(HERE/'fit.json',dict(gate='insufficient_support',evidence=evidence,total=total))
        return
    d=direction(episodes)
    delta_logits=np.asarray([np.sum(d*x) for x in diffs])
    scale=min(1.,.6/max(np.linalg.norm(d),1.e-15),.45/max(np.max(np.abs(delta_logits)),1.e-15))
    delta=(scale*d).astype(np.float32)
    cos={}
    for seed in SEEDS:
        v=direction([r for r in episodes if r['seed']!=seed]);cos[seed]=float(np.sum(v*d)/max(1.e-15,np.linalg.norm(v)*np.linalg.norm(d)))
    sys.path.insert(0,str(ACTOR/'blended_actor_dev/serving'))
    import importlib.util
    spec=importlib.util.spec_from_file_location('signed_sapling_packer',ACTOR/'blended_actor_dev/serving/build.py')
    pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
    weights=FIT/'candidate.npz'; target=HERE/'candidate.npz'
    with np.load(weights,allow_pickle=False) as z:
        original_embedded=pack.quantize(weights,z)
        if original_embedded!=pack.literal(PARENT.read_text(),pack.STUDENT):raise ValueError('parent export does not exactly reproduce deployed student')
        params={k:z[k] for k in z.files}
        params['whead.weight']=params['whead.weight']+delta[:,:96]
        params['whead.bias']=params['whead.bias']+delta[:,96]
        np.savez(target,**params)
        embedded=pack.quantize(target,z)
    src=PARENT.read_text();out=HERE/'candidate.py';out.write_text(pack.substitute(src,embedded))
    if pack.substitute(out.read_text(),pack.literal(src,pack.STUDENT))!=src or out.stat().st_size>10000000 or sha(out)==sha(PARENT):
        raise ValueError('deployment differs outside student tensors, oversized or no change')
    held=[replay(c,newer=out) for c in p['held_command_audit']]
    write(HERE/'held_commands.json.gz',held)
    seat_changes={s:sum(v['changed'] for g in held if g['seat']==s for v in g['rows']) for s in (0,1)}
    good=all(seat_changes.values()) and sum(v>0 for v in cos.values())>=2
    write(HERE/'fit.json',dict(gate='eligible_for_old_root_response' if good else 'stop_at_offline_gate', protocol=sha(HERE/'protocol.json'),
        score_receipt=sha(HERE/'scores.json'), selected=sha(out),weights=sha(target),file_bytes=out.stat().st_size,
        actual_supported_decisions=total, physical_alternatives_selected={s:sum(v) for s,v in choices.items()}, episodes=evidence,
        leave_one_map_out_cosine=cos, scale=scale, step_l2=float(np.linalg.norm(delta)),max_abs_paired_logit_shift=float(np.max(np.abs(scale*delta_logits))),
        parameter_count=int(np.count_nonzero(delta)),held_public_command_changes=seat_changes,held=sha(HERE/'held_commands.json.gz')))
    print(read(HERE/'fit.json')['gate'],seat_changes,flush=True)
if __name__=='__main__':run()

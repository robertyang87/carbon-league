"""Recover explicit common decoder uniforms behind the actual sampled team commands on all signed training roots."""
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

import numpy as np
from run import HERE, background, policy, read, snapshots, write


def pushed_intervals(root,slot):
    m=root.m
    p=[]
    for residual in (False,True):
        _,r=root.decode(residual,slot)
        p.append(r[slot])
    cuts=[np.cumsum(r) for r in p]
    for c in cuts:c[-1]=1.
    ends=sorted({0.,1.,*(float(v) for c in cuts for v in c[:-1] if 0<v<1)})
    groups=defaultdict(list)
    def issued(z):
        pair=[root.decode(i==1,slot,min(4,int(np.searchsorted(c,z,side='right'))))[0] for i,c in enumerate(cuts)]
        chosen,rejected=policy.guard(m,root.x,*pair)
        return policy.key(root.f.to_actions(chosen)),bool(rejected)
    for low,high in zip(ends,ends[1:]):
        k,rollback=issued(low)
        groups[k].append((low,high,rollback))
    return groups,issued


def main():
    module=policy.load()
    plan=read(HERE/'map_declaration.json')
    result=[]
    for i,seed in enumerate(plan['training']):
        for seat in (0,1):
            observed=read(HERE/'training'/f'{seed}_{seat}.json.gz')
            r=observed['root']
            winner,*_=snapshots(module,seed,seat,r['opponent'])
            _,t,slot,_,root,*_=winner
            assert (t,slot)==(r['t'],r['slot'])
            model=root.distribution(slot,policy.ZERO)
            groups,execute=pushed_intervals(root,slot)
            assert set(groups)==set(model)
            for k,sub in groups.items():assert abs(sum(hi-lo for lo,hi,rejected in sub)-model[k][0])<1e-12
            names=sorted(model);prefix=np.r_[0.,np.cumsum([model[k][0] for k in names])]
            proof=[]
            for uniform,k in zip(r['uniforms'],r['draws']):
                ofs=uniform-prefix[names.index(k)]
                for lo,hi,rolled in groups[k]:
                    if ofs<hi-lo:
                        z=lo+ofs
                        assert execute(z)==(k,rolled)
                        proof.append(dict(sample_uniform=uniform,common_decoder_uniform=float(z),guard_rollback=rolled,
                                          physical_joint_command=k));break
                    ofs-=hi-lo
                else:raise ValueError('failed to locate sampled probability mass')
            errors=[]
            for j in range(5):
                h=np.zeros(5,np.float32);h[j]=.001
                pos=root.distribution(slot,h);neg=root.distribution(slot,-h)
                same=set(model)==set(pos)==set(neg)
                if same:
                    errors.append(max(abs((pos[k][0]-neg[k][0])/.002-model[k][1][j]) for k in model))
                else:
                    errors.append(None)
            scores=np.asarray([observed['outcomes'][k]['official']['u_by_seat'][seat] for k in r['draws']])
            advantages=(8*scores-scores.sum())/7
            gradient=np.mean([a*model[k][1]/model[k][0] for a,k in zip(advantages,r['draws'])],axis=0)
            result.append(dict(seed=seed,seat=seat,t=t,slot=slot,draws=proof,source_sha256=r['source_sha256'],
                               root_gradient=gradient.tolist(),root_gradient_l2=float(np.linalg.norm(gradient)),
                               executed_support=len(names),by_bias_max_abs_centered_error=errors,
                               guard_rollbacks=sum(d['guard_rollback'] for d in proof)))
    write(HERE/'likelihood_evidence.json.gz',dict(roots=result,executed_joint_commands_verified=sum(len(q['draws']) for q in result),
          method='Category sampling on complete issued commands equals shared-uniform decoding. For each observed categorical draw replay its within-category rank across the original two decoder CDF intervals; both passes and the production guard recover that exact physical team order.'))
    print('verified',len(result),'roots',sum(len(q['draws']) for q in result),'commands',
          'max finite-difference error',max(e for r in result for e in r['by_bias_max_abs_centered_error'] if e is not None))

if __name__=='__main__':main()

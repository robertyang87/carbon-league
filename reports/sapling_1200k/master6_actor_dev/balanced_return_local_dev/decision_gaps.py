"""Quantify how close the learned real action logits came to changing the greedy action at the 32 anchors."""
from pathlib import Path
import importlib.util
import numpy as np
from preregister import HERE, PREV, read, save, sha

spec=importlib.util.spec_from_file_location('local_independent_mechanism',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def runner_gap(logits, legal, incumbent):
    rivals=np.flatnonzero(legal)
    rivals=rivals[rivals!=incumbent]
    if not len(rivals):
        raise ValueError('registered focus has no alternative')
    runner=int(rivals[np.argmax(logits[rivals])])
    return runner,float(logits[incumbent]-logits[runner])


def main():
    audit.common.assert_cpu()
    receipt=read(HERE/'mechanism.json')
    assert receipt['audit_sha256']==sha(HERE/'audit.py') and receipt['passed']
    parent=audit.common.policy.load(audit.common.PARENT)
    with np.load(PREV/'candidate.npz') as n:
        learned=np.ascontiguousarray(n['whead.weight'].T),n['whead.bias']
    rows=[]
    for r in receipt['rows']:
        t=r['step']
        if t not in (205,271) or r['focal_slot']<0:
            continue
        seed,seat,focal=r['seed'],r['physical_seat'],r['focal_slot']
        world,features,prefix,source=audit.common.open_state(seed,r['original_seat'],t)
        f=features[seat];x=f.observe(world.obs(seat));enc=audit.encoded(parent,x)
        assert f.own_uid[focal]==r['focal_uid']
        for residual in (False,True):
            _,old=audit.independent_decode(parent,x,f.aux,enc,-1,residual,learned)
            _,new=audit.independent_decode(parent,x,f.aux,enc,focal,residual,learned)
            a,b=old[focal],new[focal];legal=a['legal'];act=int(a['action'])
            assert np.array_equal(legal,b['legal'])
            old_runner,old_gap=runner_gap(a['combined'],legal,act)
            new_runner,new_gap=runner_gap(b['combined'],legal,act)
            perturb=b['combined'][legal]-a['combined'][legal]
            rows.append(dict(seed=seed,seat=seat,step=t,uid=r['focal_uid'],residual=residual,
                         parent_command=act,candidate_command=int(b['action']),old_best_alternative=old_runner,
                         new_best_alternative=new_runner,parent_winning_gap=old_gap,
                         candidate_gap_over_best_rival=new_gap,relative_change_amplitude=float(np.ptp(perturb))))
    assert len(rows)==64
    def summary(items):
        return dict(count=len(items),changed_argmax=sum(x['parent_command']!=x['candidate_command'] for x in items),
                    incumbent_gap_decreased=sum(x['candidate_gap_over_best_rival']<x['parent_winning_gap'] for x in items),
                    median_parent_gap=float(np.median([x['parent_winning_gap'] for x in items])),
                    median_candidate_gap=float(np.median([x['candidate_gap_over_best_rival'] for x in items])),
                    min_candidate_gap=min(x['candidate_gap_over_best_rival'] for x in items),
                    median_relative_change_amplitude=float(np.median([x['relative_change_amplitude'] for x in items])))
    save(HERE/'decision_gaps.json',dict(mechanism_sha256=sha(HERE/'mechanism.json'),script_sha256=sha(HERE/'decision_gaps.py'),
         scope='Post-registration diagnostic before any farther tuning. Each branch is separate; 64 branches are not 64 independent roots.',
         basic=summary([x for x in rows if not x['residual']]),return_adjusted=summary([x for x in rows if x['residual']]),rows=rows))


if __name__=='__main__':
    main()

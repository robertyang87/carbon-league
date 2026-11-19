"""Show the real decoder can change an issued command on a previously seen neighbor when its timing gate is opened in memory."""
import importlib.util
import numpy as np
from preregister import HERE, PREV, read, save, sha
spec=importlib.util.spec_from_file_location('local_independent_probe',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def verify():
    audit.common.assert_cpu()
    receipt=read(HERE/'mechanism.json')
    cases=[r for r in receipt['rows'] if r['focal_slot']<0 and r['as_if_different_joint_command']]
    rows=[]
    parent=audit.common.policy.load(audit.common.PARENT)
    cand=audit.common.policy.load(HERE/'candidate.py')
    with np.load(PREV/'candidate.npz') as npz:
        head=(np.ascontiguousarray(npz['whead.weight'].T),npz['whead.bias'].copy())
    for r in cases:
        seed,t,seat,orig,anchor,focal=(r[k] for k in ('seed','step','physical_seat','original_seat','reference_window','as_if_focus'))
        w,ff,_,src=audit.common.open_state(seed,orig,t);f=ff[seat]
        x=f.observe(w.obs(seat));enc=audit.encoded(parent,x);aux=f.aux
        baseline=f.to_actions(parent._mixed_actions(x,aux));actual=f.to_actions(cand._mixed_actions(x,aux))
        assert baseline==actual
        old_slot,old_windows=cand._local_slot,cand._LOCAL_WINDOWS
        cand._LOCAL_WINDOWS=(*old_windows,t)
        cand._local_slot=lambda x,aux,enc,clock:old_slot(x,aux,enc,anchor if clock==t else clock)
        try:
            with audit.capture_decoder(cand) as passes:
                observed=f.to_actions(cand._mixed_actions(x,aux))
            decode=[];subsequent=[]
            for p in passes:
                if p['entropy_probe']:
                    continue
                assert p['focal']==focal
                residual=p['residual']
                y,trace=audit.independent_decode(parent,x,aux,enc,focal,residual,head)
                _,p0=audit.independent_decode(parent,x,aux,enc,-1,residual,head)
                audit.same_trace(trace,p['steps']);decode.append(y)
                for i in trace:
                    if i>focal and (trace[i]['previous']!=p0[i]['previous'] or not np.array_equal(trace[i]['move'],p0[i]['move']) or not np.array_equal(trace[i]['unit'],p0[i]['unit'])):
                        subsequent.append([residual,i])
            expect,rolled=audit.common.policy.guard(parent,x,*decode)
            assert f.to_actions(expect)==observed and observed!=actual
            rows.append(dict(seed=seed,source_sha256=src['sha256'],seat=seat,t=t,anchor=anchor,uid=f.own_uid[focal],
                             physical_slot=focal,actually_issued_by_registered_policy=actual,
                             diagnostic_command=observed,guard_rollback=bool(rolled),
                             verified_later_slots_with_different_prefix=subsequent))
        finally:
            cand._local_slot,cand._LOCAL_WINDOWS=old_slot,old_windows
        assert f.to_actions(cand._mixed_actions(x,aux))==actual
    save(HERE/'positive_control.json', dict(source_sha256=sha(HERE/'mechanism.json'),
         purpose='All previously identified neighboring as-if differences, selected retrospectively; only in-memory eligibility was mapped to nearest anchor. No match or counterfactual terminal payoff.',
         code_sha256=sha(HERE/'positive_control.py'),cases=rows))


if __name__=='__main__':
    verify()

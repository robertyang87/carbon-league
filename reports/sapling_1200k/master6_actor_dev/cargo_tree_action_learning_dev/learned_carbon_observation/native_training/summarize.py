"""Freeze first eligible roots without terminal results, then inspect local input support and prior branch identity."""
from collections import Counter,defaultdict
import gzip
from pathlib import Path
import sys
import numpy as np

from scan import HERE,LEARN,ROOT,F,REG,PACKS,public_frames,read,save,sha,compact
sys.path.insert(0,str(LEARN/'action_preference'))
from support import option
from runtime import _carbon_view

def pick(book,reg):
    by={d['card']['terrain']:d for d in book};pilot=[]
    for h in reg['terrain_priority']:
        d=by[h];choices=[]
        for seat in d['seats']:
            s=seat['seat']
            for ev in seat['events']:
                if ev['eligible_workers']:
                    choices.append((ev['t'],sha(f"{reg['tag']}|{h}|{s}".encode()),s,ev));break
        if not choices:continue
        _,_,s,ev=min(choices,key=lambda x:x[:2]);pilot.append(dict(card=d['card'],seat=s,t=ev['t'],public_sha256=ev['visible_sha256'],
            parent=ev['parent'],student=ev['student'],economic_workers=ev['eligible_workers'],archived_own=ev['recorded_own'],
            root_id=sha(compact([h,s,ev['t'],ev['visible_sha256'],ev['parent'],ev['student']]))))
        if len(pilot)==reg['max_pilot_terrains']:break
    return pilot

def local_card(d,net):
    events={(s['seat'],e['t']):e for s in d['seats'] for e in s['events']};fs=[F.Feat(s) for s in (0,1)]
    out=[]
    for states,_,orders,cfg in public_frames(d['card']):
        for s,o in enumerate(states):
            t=o['step'];f=fs[s];x=f.observe(o)
            ev=events.get((s,t))
            if ev is None:continue
            assert sha(compact(o))==ev['visible_sha256']
            old=x['maps'][0].astype(np.float64)*(100/255);new=_carbon_view(x,f,F,net)['maps'][0].astype(np.float64)*(100/255)
            changed=(old!=new).reshape(225);old=old.reshape(225);new=new.reshape(225);units=[]
            for u in ev['changed']:
                own=o['players'][s];commands=[ev[n].get(u,'NONE') for n in ('parent','student')]
                values=[];union=set()
                for board in (old,new):
                    col=[]
                    for cmd in commands:
                        v,fp=option(board,own[1],own[2],u,cmd,F.CANON if s else F.GRID)
                        col.append(v);union.update(fp)
                    values.append(np.stack(col))
                a,b=values;delta=(b[1]-b[0])-(a[1]-a[0])
                units.append(dict(uid=u,role=ev['changed'][u]['role'],economic_eligible=u in ev['eligible_workers'],commands=commands,
                    old=a.tolist(),learned=b.tolist(),per_channel_contrast_delta=delta.tolist(),max_contrast=float(np.abs(delta).max()),
                    max_either_arm=float(np.abs(b-a).max()),changed_footprint_cells=int(changed[list(union)].sum())))
            econ=[r for r in units if r['economic_eligible']]
            out.append(dict(seed=d['card']['seed'],terrain=d['card']['terrain'],seat=s,t=t,phase=ev['phase'],changed_map_cells=int(changed.sum()),
                whole_joint_max_contrast=max(r['max_contrast'] for r in units),
                economic_worker_max_contrast=max((r['max_contrast'] for r in econ),default=None),units=units))
    assert len(out)==len(events)
    return out

def prior_pairs(cases):
    exp=LEARN.parent.parent;base=exp.parent
    prior=[];registry=exp/'joint_response_critic_dev/signed_roots.json.gz'
    for r in read(registry)['rows']:
        prior.append((r['seed'],r['seat'],r['t'],r['parent'],r['alternative'],r.get('source_sha256'),r['case']))
    nregistry=len(prior);nfiles=0
    for f in sorted(base.rglob('live_parent.json')):
        f2=f.with_name('live_alternative.json')
        if not f2.exists():continue
        a,b=read(f),read(f2);root=a.get('root')
        if not isinstance(root,dict) or 't' not in root:continue
        candidate=None;incumbent=None
        for ticket,name in ((a,'parent'),(b,'alternative')):
            tape=f.with_name('live_'+name+'_actions.json.gz')
            if tape.exists():
                seq=read(tape)
                if seq and isinstance(seq[0],list):
                    if name=='parent':incumbent=seq[0][a['seat']]
                    else:candidate=seq[0][a['seat']]
        if incumbent is None or candidate is None:continue
        nfiles+=1;prior.append((a['seed'],a['seat'],root['t'],incumbent,candidate,None,str(f.relative_to(ROOT))))
    native=read(LEARN/'native_response/result.json')
    for c in native['maps']:
        if c['changed']:
            a=c['arms']['parent'];b=c['arms']['student'];s=c['seat']
            prior.append((c['seed'],s,c['root_t'],a['root_orders'][s],b['root_orders'][s],None,'learned_carbon_observation/native_response'))
    out=[]
    for c in cases:
        hits=[r for r in prior if (r[0],r[1],r[2])==(c['card']['seed'],c['seat'],c['t'])]
        exact=[r for r in hits if (r[3],r[4]) in ((c['parent'],c['student']),(c['student'],c['parent']))]
        out.append(dict(root_id=c['root_id'],same_seed_seat_step=len(hits),same_both_joint_orders=len(exact),
            references=[r[-1] for r in hits],source_sha_matches=sum(r[5]==c['card']['sha256'] for r in hits)))
    return dict(prior_signed_rows=nregistry,archived_two_arm_receipt_directories=nfiles,prior_current_candidate_native_cases=len(native['maps']),flags=out,
                scope='Signed registry, paired live_parent/live_alternative archived action tapes, and both current native-response cases. Distinct from per-model held action probes. These sources overlap; counts of prior files/rows are not unique-root totals.')

def summary():
    reg=read(REG);cards=reg['training_cards'];book=[read(HERE/(str(c['seed'])+'.json.gz')) for c in cards]
    assert all(v['card']==c and v['registration']==sha(REG.read_bytes()) for v,c in zip(book,cards))
    pilot=pick(book,reg)
    save(HERE/'pilot.json',dict(registration=sha(REG.read_bytes()),source='archived public states',selection_uses_local_field=False,roots=pilot))
    dup=prior_pairs(pilot);save(HERE/'duplicates.json',dup)
    with np.load(LEARN/'weights.npz') as z:weights=[z[k] for k in 'wbva']
    local=[r for d in book for r in local_card(d,weights)]
    save(HERE/'local_contrasts.json.gz',local)
    lu={(r['seed'],r['seat'],r['t']):r for r in local}
    def support(roots):
        worker=[r for r in roots if r['economic_worker_max_contrast'] is not None]
        return dict(all_changed_turns=len(roots),economic_worker_turns=len(worker),any_joint_contrast=sum(r['whole_joint_max_contrast']>1e-6 for r in roots),
            any_economic_worker_contrast=sum(r['economic_worker_max_contrast']>1e-6 for r in worker),
            economic_ge_one_carbon=sum(r['economic_worker_max_contrast']>=1 for r in worker),economic_board_changed=sum(r['changed_map_cells']>0 for r in worker))
    per=[];g=Counter();ph=defaultdict(Counter);roles=Counter();tags=Counter();surfaces=Counter();cargo=Counter();ps=[]
    for d in book:
        for s in d['seats']:
            cc=s['counts'];g.update(cc)
            for k,v in s['stage'].items():ph[k].update(v)
            roles.update(s['changed_roles']);first_all=next((e['t'] for e in s['events']),None)
            first_econ=next((e['t'] for e in s['events'] if e['eligible_workers']),None)
            per.append(dict(seed=d['card']['seed'],terrain=d['card']['terrain'],seat=s['seat'],counts=cc,first_any_change=first_all,
                first_economic_change=first_econ,stage=s['stage'],changed_roles=s['changed_roles'],local=support([lu[d['card']['seed'],s['seat'],e['t']] for e in s['events']])))
            for e in s['events']:
                for u in e['eligible_workers']:
                    v=e['changed'][u];tags.update(v['different']);surfaces[v['surface']]+=1
                    if v['role']=='COLLECTOR':cargo['loaded' if v['cargo']>0 else 'empty']+=1
    prev=read(LEARN/'native_response/result.json');pair=next(c for c in prev['maps'] if c['changed'])
    # Measured original suffixes include environment, parent reconstruction and all calls. Only a planning estimate.
    sec_per_paired_joint=sum(a['whole_arm_elapsed_s']/(a['transitions']-a['root']) for a in pair['arms'].values())
    for c in pilot:
        a=lu[c['card']['seed'],c['seat'],c['t']]
        ps.append(dict(root_id=c['root_id'],seed=c['card']['seed'],seat=c['seat'],t=c['t'],economic_max_carbon=a['economic_worker_max_contrast'],
            whole_joint_max_carbon=a['whole_joint_max_contrast'],changed_board_cells=a['changed_map_cells'],units=a['units'],
            paired_suffix_cost_min=(299-c['t'])*sec_per_paired_joint/60))
    all_econ=[r for r in local if r['economic_worker_max_contrast'] is not None]
    res=dict(protocol=sha(REG.read_bytes()),pilot=sha((HERE/'pilot.json').read_bytes()),duplicates=sha((HERE/'duplicates.json').read_bytes()),
        local_source_sha256=sha((LEARN/'action_preference/support.py').read_bytes()),local_rows_sha256=sha((HERE/'local_contrasts.json.gz').read_bytes()),
        sources={str(c['seed']):sha((HERE/(str(c['seed'])+'.json.gz')).read_bytes()) for c in cards},totals=g,by_phase=ph,changed_roles=roles,
        economic_reasons=tags,root_surfaces=surfaces,collector_cargo=cargo,seats=per,overall_local=support(local),pilot_local=ps,
        zero_maps=[c['seed'] for c in cards if not any(r['seed']==c['seed'] for r in all_econ)],actual_time_range=[min(r['t'] for r in all_econ),max(r['t'] for r in all_econ)] if all_econ else [],
        paired_wall_seconds_per_remaining_joint=sec_per_paired_joint,native_live_calls=0)
    save(HERE/'summary.json',res);return res
if __name__=='__main__':summary()

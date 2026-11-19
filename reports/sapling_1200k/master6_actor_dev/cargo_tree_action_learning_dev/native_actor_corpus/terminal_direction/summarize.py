"""Aggregate coverage before fitting. Terminal payoffs are defined only for truly matched decisions."""
from collections import Counter,defaultdict
from compare import HERE,read,sha,write,np
def summary(rows,name):
    v=[(r,r['by_scalar'][name]) for r in rows]
    changed=[(r,z) for r,z in v if z['changed']]
    both=[(r,z) for r,z in changed if z['category']=='both_observed']
    bymap=defaultdict(list)
    for r,z in both:bymap[r['terrain']].append((z['paired_wdl'],z['paired_official_margin']))
    map_means={k:np.mean(z,axis=0).round(6).tolist() for k,z in bymap.items()}
    return dict(records=len(rows),terrains=len({r['terrain'] for r in rows}),distinct_public_roots=len({(r['prefix_sha256'],r['public_sha256'],r['seat']) for r in rows}),
         unchanged=len(v)-len(changed),changed=len(changed),changed_maps=len({r['terrain'] for r,z in changed}),
         center_command_observed=sum(z['center_archived_arm'] is not None for r,z in v),variant_command_observed=sum(z['choice_archived_arm'] is not None for r,z in v),
         both_observed_changes=len(both),both_observed_change_maps=len(bymap),changed_unmeasured=len(changed)-len(both),
         both_changed_archival_wdl=sum(z['paired_wdl']!=0 for r,z in both),map_mean_on_supported_changes=map_means,
         supported_changes_only_avg=np.mean(list(map_means.values()),axis=0).round(6).tolist() if map_means else None)
def pooled_support(rows,names):
    # Two different experiments that share a public root may offer a third action.
    # A different random draw on each side does not make a new common-stream pair.
    by=defaultdict(list)
    for r in rows:by[r['continuation'],r['prefix_sha256'],r['public_sha256'],r['seat']].append(r)
    result={}
    for name in names:
        same_root_only=[];direct=[]
        for roots in by.values():
            r=roots[0];center=r['free_commands']['center'];variant=r['free_commands'][name]
            assert all(z['free_commands']['center']==center and z['free_commands'][name]==variant for z in roots)
            if center==variant:continue
            if any(z['by_scalar'][name]['category']=='both_observed' for z in roots):
                direct.append((r['terrain'],r['public_sha256']))
            elif all(any(a in z['archived_commands'] for z in roots) for a in (center,variant)):
                same_root_only.append(dict(seed=r['seed'],seat=r['seat'],t=r['t'],continuation=r['continuation'],cases=[z['case'] for z in roots]))
        result[name]=dict(same_stream_measured_changed_distinct_roots=len(direct),same_stream_measured_changed_boards=len({t for t,_ in direct}),
                          both_measured_only_in_separate_experiments=same_root_only)
    return result

def main():
    protocol=read(HERE/'protocol.json');rr=read(HERE/'commands.json.gz')
    names=[k for k in protocol['scalars'] if k!='center'];groups=defaultdict(list)
    for r in rr:
        groups['all_including_repeats'].append(r)
        groups['source/'+r['group']].append(r)
        if r.get('repeat'):continue
        groups['all_unique_contrasts'].append(r)
        groups['suffix/'+r['continuation']].append(r)
        groups['modelterrain/'+r['native_model_terrain_split']].append(r)
        if r['continuation']==protocol['freeze_parent']:
            groups['primary'].append(r)
            groups['primary/'+r['native_model_terrain_split']].append(r)
    primary=groups['primary'];train=[r for r in primary if r['native_model_terrain_split']=='train']
    stats={k:{n:summary(rows,n) for n in names} for k,rows in groups.items()}
    support={n:summary(train,n) for n in names}
    eligible=[n for n,p in support.items() if p['both_observed_change_maps']>=2]
    fit=None
    if eligible:
        # Compare eligible scalars on a common, fully evaluable population.
        # Missing chosen command is never silently assigned the parent's return.
        common=[r for r in train if all(r['by_scalar'][n]['category']!='changed_unmeasured' for n in eligible)]
        pool={n for n in eligible if summary(common,n)['both_observed_change_maps']>=2}
        if pool:
            options=['center',*sorted(pool)]
            returns={}
            for n in options:
                b=defaultdict(list)
                for r in common:
                    z=r['by_scalar'][n];b[r['terrain']].append((z['paired_wdl'],z['paired_official_margin']))
                returns[n]=np.mean([np.mean(v,axis=0) for v in b.values()],axis=0).tolist()
            choice=max(options,key=lambda n:(*returns[n],-abs(protocol['scalars'][n])))
            fit=dict(rows=len(common),maps=len({r['terrain'] for r in common}),candidates=returns,selected=choice,
                     retained_common_support=[r['case'] for r in common],
                     independent_model_held={s:summary([r for r in primary if r['native_model_terrain_split']==s],choice) for s in ['val','test']})
    calibrate=read(HERE/'training_coverage.json.gz')['rows']
    calibration={}
    for n in names:
        hits=[r for r in calibrate if r['free_commands'][n]!=r['free_commands']['center']]
        byrole=Counter()
        for r in hits:byrole.update(r['changed_roles'][n])
        calibration[n]=dict(rows=len(calibrate),changed=len(hits),terrains=len({r['terrain'] for r in hits}),roles=dict(byrole),
                            per_seed={str(s):sum(r['seed']==s for r in hits) for s in sorted({r['seed'] for r in calibrate})})
    sources={r['registry']:True for r in rr}
    from compare import ROOT
    write(HERE/'result.json',dict(protocol_sha256=sha(HERE/'protocol.json'),packages_sha256=sha(HERE/'packages.json'),
                inference_sha256=sha(HERE/'commands.json.gz'),exclusions_sha256=sha(HERE/'exclusions.json.gz'),training_coverage_sha256=sha(HERE/'training_coverage.json.gz'),
                source_registries={s:sha(ROOT/s) for s in sources},coverage=stats,training_coverage=calibration,
                root_level_union_support=pooled_support(rr,names),primary_training=support,eligible_scalars=eligible,fit=fit,decision='RETAIN_ON_INSUFFICIENT_SUPPORT' if fit is None else 'OFFLINE_FIT_ONLY'))
    print('primary',[(n,v['records'],v['changed'],v['both_observed_changes'],v['both_observed_change_maps']) for n,v in stats['primary'].items()])
    print('eligible',eligible,'calibration',[(n,d['changed']) for n,d in calibration.items()])
if __name__=='__main__':main()

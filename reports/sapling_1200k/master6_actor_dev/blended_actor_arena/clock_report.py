"""Cost and adaptive-search completion at the processor allocation actually used for the eight games."""
from collections import Counter,defaultdict
import statistics
import sys
from shared import HERE,read,require,save,sha


def summarize(partial=False):
    binding=read(HERE/'games/binding.json')
    games=[];per_arm=defaultdict(list);windows=defaultdict(list)
    for mi,ss in enumerate(binding['boards']):
        s=int(ss)
        for seat in (0,1):
            for ai,arm in enumerate(('parent','candidate')):
                p=HERE/'games'/arm/f'wall_{s}_{seat}.json.gz'
                if partial and not p.exists():continue
                w=read(p);calls=w['calls'];n=len(calls);cpu=(mi+seat+ai)%2
                require(w['identity']['cpu']==cpu and n>0 and len(w['segments'])>0,'processor/call mismatch')
                rootcpu=sum(x['process_cpu_seconds'] for x in calls);rootwall=sum(x['public_wall_seconds'] for x in calls)
                kinds=Counter(x['kind'] for x in calls);depths=Counter(x['max_depth'] for x in calls)
                host=[]
                for seg in w['segments']:
                    a,b=seg['start'],seg['end'];elapsed=b['monotonic_seconds']-a['monotonic_seconds'];proc=b['worker_cpu_seconds']-a['worker_cpu_seconds']
                    tick=[h-l for l,h in zip(a['host_cpu_ticks'],b['host_cpu_ticks'])];unit=a['ticks_per_second']
                    require(elapsed>0 and proc>0 and unit>0,'resource clocks')
                    host.append({'cpu':cpu,'elapsed_seconds':elapsed,'worker_cpu_seconds':proc,'core_busy_excluding_idle_iowait_steal_seconds':sum(tick[i] for i in (0,1,2,5,6))/unit,
                                 'host_steal_seconds':tick[7]/unit,'whole_cgroup_throttled_seconds':(b['cgroup_cpu']['throttled_usec']-a['cgroup_cpu']['throttled_usec'])/1e6,
                                 'whole_cgroup_throttles':b['cgroup_cpu']['nr_throttled']-a['cgroup_cpu']['nr_throttled']})
                    windows[s,seat].append([a['monotonic_seconds'],b['monotonic_seconds']])
                row={'seed':s,'seat':seat,'arm':arm,'cpu':cpu,'turns':n,'root_wall_seconds':rootwall,'root_cpu_seconds':rootcpu,
                     'mean_original_public_call_ms':1000*rootwall/n,'mean_max_depth':sum(k*v for k,v in depths.items())/n,
                     'completed_depth_counts':dict(depths),'search_return_counts':dict(kinds),'native_aborted':w['summary']['native_aborted'],
                     'original_public_above_058':w['summary']['public_seconds_over_budget'],'resource_segments':host,'wall_sha256':sha(p)}
                games.append(row);per_arm[arm].append(row)
    byarm={}
    for arm,rs in per_arm.items():
        kinds=Counter();depths=Counter()
        for r in rs:kinds.update(r['search_return_counts']);depths.update(r['completed_depth_counts'])
        n=sum(depths.values());seconds=sum(r['root_wall_seconds'] for r in rs)
        byarm[arm]=dict(turns=n,search_return_counts=dict(kinds),depth_counts=dict(depths),mean_max_depth=sum(k*v for k,v in depths.items())/n,
                        mean_original_public_call_ms=1000*seconds/n,original_public_wall_seconds=seconds,
                        original_cpu_seconds=sum(r['root_cpu_seconds'] for r in rs),both_agent_and_arena_worker_cpu_seconds=sum(g['worker_cpu_seconds'] for r in rs for g in r['resource_segments']))
    wave_seconds={f'{s}/{seat}':max(x[1] for x in ranges)-min(x[0] for x in ranges) for (s,seat),ranges in windows.items()}
    doc={'games':games,'by_arm':byarm,'elapsed_range_of_recorded_original_process_segments_by_pair_seconds':wave_seconds,
         'sum_of_four_wave_windows_seconds':sum(wave_seconds.values()),'clock_source_sha256':sha(HERE/'clock.py'),
         'caveats':['worker CPU includes both agents, simulation, checkpoints and any prefix replay; search CPU counts only measured original public calls',
                    'one observation per decision, complete depths as natively reported; single branch and depth-zero are distinguished',
                    'wave intervals span the earliest to the latest recorded game segment; external start-up, final proof and other jobs between waves are excluded',
                    'whole-cgroup counters are shared and cannot be summed across concurrent workers; host occupancy can also include processes outside this cgroup']}
    if not partial:
        require(len(games)==8 and len(windows)==4,'full panel not yet complete')
        save(HERE/'clock_summary.json',doc)
    return doc


if __name__=='__main__':
    result=summarize('--partial' in sys.argv);print([(a,b['turns'],b['mean_max_depth'],b['mean_original_public_call_ms']) for a,b in result['by_arm'].items()])

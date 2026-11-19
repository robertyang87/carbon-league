"""Four new opening-adapter runs against unmodified live Master3s2. Keep four older controls as history."""
import fcntl
import gzip
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ARENA=HERE.parent/'blended_actor_arena'
sys.path.insert(0,str(ARENA))
from shared import ACTOR,BASE,ROOT,git,module,read,require,save,sha
sys.path.insert(0,str(BASE))
import clock
from shared import TRAIN
BASELINE=TRAIN/'serving/candidate.py'
NEW=TRAIN/'opening_training'
OUT=HERE/'games';CTRL=ARENA/'games';PLAN=HERE/'protocol.json'
ORIGINAL_REUSE=ACTOR/'tree_tenure_dev/pay_policy/native_run.py'
SHARED_LOCK=ACTOR/'tree_response_training/.continuation.lock'


def pack():
    p=read(PLAN)
    cache=Path(tempfile.gettempdir())/('sapling-only-opening-'+p['candidate_pack_sha256']+'.py')
    if cache.exists():
        require(sha(cache)==p['candidate_pack_sha256'],'wrong cached single file')
        return cache
    with tempfile.TemporaryDirectory(dir=cache.parent) as folder:
        tmp=Path(folder);dest=tmp/'candidate.py';build=module(NEW/'build.py','adapter_builder')
        for name in ('protocol.json','fit.json','opening_weights.npz'):shutil.copyfile(NEW/name,tmp/name)
        build.HERE=tmp;rebuilt=build.build(dest)
        require(rebuilt==read(NEW/'release.json') and sha(dest)==p['candidate_pack_sha256'],'materialized release')
        os.replace(dest,cache)
    return cache


def driver(out,new_path):
    d=module(ORIGINAL_REUSE,'opening_official_original')
    d.OUT=out;d.CANDIDATE=new_path;d.PARENT=BASELINE
    return d


def historical(row,src_auth):
    for doc in row['files'].values():require(sha(ROOT/doc['path'])==doc['sha256'],'historical tape or clock changed')
    old=driver(CTRL,BASELINE);s=row['seed'];seat=row['seat']
    check=clock.verify_completed(old,('candidate',s,seat,row['cpu'],row['terrain'],src_auth))
    signed=read(ROOT/row['files']['receipt']['path']);require(signed['official']==old.official.check(CTRL/'candidate',signed['game']),'historical official re-accounting')
    require(check['identity']['binding']['candidate']==sha(BASELINE) and check['identity']['cpu']==row['cpu'],'historical package/core')
    return signed


def admit():
    proposal=read(PLAN);require(sha(PLAN)=='20e857e0f535b0fb6becf26326847341979431ed99865238fb1d56c2c13d48ab','registration')
    require(sha(ARENA/'verification.json')==proposal['prior_development_result_sha256'] and sha(BASELINE)==proposal['baseline_sha256'],'old comparison changed')
    for name,h in proposal['admission']['registrations'].items():require(sha(ROOT/name)==h,'opening training identity '+name)
    result=read(NEW/'normal_entry.json.gz')
    require(result['decision']=='PASS' and all(all(g.values()) for g in result['gate_components'].values()) and sum(result['changed_new_by_seat'].values())==28
            and all(result['changed_new_by_seat'].values()) and result['nonopening_probes']>0 and result['nonopening_differences_vs_frozen']==result['off_at_t0_changed']==0
            and result['release']==read(NEW/'release.json') and result['protocol_sha256']==sha(NEW/'protocol.json'),'quality gate')
    require(sha(pack())==proposal['candidate_pack_sha256'] and sha(NEW/'opening_weights.npz')==proposal['candidate_head_sha256'],'wrong published adapter')
    old_auth=read(CTRL/'binding.json')
    for row in proposal['historical_baseline']:historical(row,old_auth)
    require(len(proposal['historical_baseline'])==proposal['new_games']==4 and {(r['seed'],r['seat']) for r in proposal['historical_baseline']}==set(map(tuple,proposal['new_order'])),'missing pair')
    tools=[Path(__file__),ARENA/'clock.py',BASE/'native_clock.py',ORIGINAL_REUSE,Path(driver(OUT,pack()).arena.__file__)]
    sources={str(p.relative_to(ROOT)):sha(p) for p in tools}
    if (OUT/'binding.json').exists():
        a=read(OUT/'binding.json');require(a['registered_sha256']==sha(PLAN) and a['checked_driver_sha256']==sources and a['input_probe_sha256']==sha(NEW/'normal_entry.json.gz'),'resume code changed')
        return a
    git('fetch','-q','origin','main');rev=git('rev-parse','origin/main').decode().strip()
    for file in (PLAN,NEW/'build.py',NEW/'protocol.json',NEW/'fit.json',NEW/'normal_entry.json.gz',NEW/'release.json',NEW/'opening_weights.npz'):
        require(git('show',rev+':'+str(file.relative_to(ROOT)))==file.read_bytes(),'not published '+str(file))
    require(proposal['original']['sha256']==sha(driver(OUT,pack()).arena.prior.SOURCE)==sha(git('show',rev+':sources/master3s2/master3s2.py')),'native examiner identity')
    a={'registered_sha256':sha(PLAN),'publication_readback':rev,'candidate':sha(pack()),'baseline':sha(BASELINE),'opening_head_sha256':sha(NEW/'opening_weights.npz'),
       'checked_driver_sha256':sources,'input_probe_sha256':sha(NEW/'normal_entry.json.gz'),'original':proposal['original']['sha256'],'old_source_binding_sha256':sha(CTRL/'binding.json'),
       'role':'development only, historical baseline and fresh unseeded natural opponent','precise_terrain':old_auth['boards']}
    save(OUT/'binding.json',a);return a


def one(job):
    return clock.run_one(driver(OUT,pack()),job)


def collect(auth):
    plan=read(PLAN);old_auth=read(CTRL/'binding.json')
    prior_path=ACTOR/'actor_partner_dev/replication/native.py'
    sys.path.insert(0,str(prior_path.parent.parent))
    econ=module(prior_path,'first_economy')
    vals=[];pairs=[];scores={a:[] for a in ('parent','candidate')};timing=[]
    for r in plan['historical_baseline']:
        s=r['seed'];seat=r['seat'];hist=historical(r,old_auth)
        new=read(OUT/'candidate'/f'receipt_{s}_{seat}.json');job=('candidate',s,seat,r['cpu'],r['terrain'],auth)
        clock.verify_completed(driver(OUT,pack()),job)
        pair={'seed':s,'seat':seat,'arms':{}}
        for name,folder,pack_file,signed in [('parent',CTRL,BASELINE,hist),('candidate',OUT,pack(),new)]:
            row=signed['game'];q=folder/'candidate';raw=read(q/row['raw_file']);w=read(q/f'wall_{s}_{seat}.json.gz')
            require(signed['official']['official_reward_by_seat']==raw['official_score_by_seat'] and w['identity']['cpu']==r['cpu'] and w['raw_sha256']==row['raw_sha256'],'timing or true scoring')
            econ.OUT=folder;econ.CAND=pack_file
            scores[name].append((seat,row['margin'],row['result']))
            vals.append(dict(seed=s,seat=seat,arm=name,raw_sha256=row['raw_sha256'],**econ.economy(row,'candidate')))
            pair['arms'][name]={'win':row['result']=='win','result':row['result'],'official_margin':row['margin'],'score_by_seat':raw['official_score_by_seat'],
                                 'steps':raw['steps'],'terminal_extinction':raw['extinct_by_seat'],'raw_sha256':row['raw_sha256'],'wall_sha256':sha(q/f'wall_{s}_{seat}.json.gz')}
            timing.append({'arm':name,'historical':name=='parent','seed':s,'seat':seat,'cpu':r['cpu'],'summary':w['summary'],'segments':w['segments'],
                'mean_depth':statistics.mean(c['max_depth'] for c in w['calls']),'mean_public_ms':1000*statistics.mean(c['public_wall_seconds'] for c in w['calls']),
                'total_original_cpu_s':sum(c['process_cpu_seconds'] for c in w['calls']),'input_clock_sha256':pair['arms'][name]['wall_sha256']})
        pair['delta_official_margin']=pair['arms']['candidate']['official_margin']-pair['arms']['parent']['official_margin'];pairs.append(pair)
    bymap={s:statistics.mean(r['delta_official_margin'] for r in pairs if r['seed']==int(s)) for s in auth['precise_terrain']}
    arm={a:dict(wins=sum(z=='win' for _,_,z in v),draws=sum(z=='draw' for _,_,z in v),mean_official_margin=statistics.mean(x for _,x,_ in v),
           seats={k:dict(wins=sum(z=='win' for t,_,z in v if t==k),mean_official_margin=statistics.mean(x for t,x,_ in v if t==k)) for k in (0,1)}) for a,v in scores.items()}
    save(HERE/'income.json',vals);save(HERE/'clock_summary.json',timing)
    save(HERE/'comparison.json',dict(completed_games=8,new_game_count=4,full_development_source_games=8,historical_baseline=True,formal_games=0,pairs=pairs,
        by_arm=arm,delta_by_terrain=bymap,paired_delta=statistics.mean(bymap.values()),cluster_se=statistics.stdev(bymap.values())/(2**.5),
        income_sha256=sha(HERE/'income.json'),clock_summary_sha256=sha(HERE/'clock_summary.json'),binding_sha256=sha(OUT/'binding.json')))


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);auth=admit();rows=read(PLAN)['historical_baseline']
        for s in dict.fromkeys(r['seed'] for r in rows):
            jobs=[('candidate',s,r['seat'],r['cpu'],r['terrain'],auth) for r in rows if r['seed']==s]
            if all((OUT/'candidate'/f'wall_{s}_{j[2]}.json.gz').exists() for j in jobs):continue
            with SHARED_LOCK.open('a') as slot:
                fcntl.flock(slot,fcntl.LOCK_EX)
                print('running registered candidate map',s,flush=True)
                with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
                    for result in pool.map(one,jobs):print(result['official'],flush=True)
        collect(auth)
if __name__=='__main__':main()

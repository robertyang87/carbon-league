"""Registered eight-map confirmation with both seats, the original examiner and atomic per-turn recovery."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import gzip
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
BASE=ACTOR.parent
ROOT=BASE.parents[1]
sys.path[:0]=[str(ACTOR/'blended_actor_arena'),str(BASE)]
from shared import CACHE,git,module,read,require,save,sha
import clock
import map_identity
from tree_win_gate import RESERVED_DEVELOPMENT_PLANS, RESERVED_RANGE
from tree_audited_game import exam
from zerosum_env import make
import numpy as np

PLAN=HERE/'protocol.json'
OUT=HERE/'games'
CAND=ACTOR/'actor_partner_dev/candidate.py'
ORIGINAL_DRIVER=ACTOR/'tree_tenure_dev/pay_policy/native_run.py'
SHARED=ACTOR/'tree_response_training/.continuation.lock'
REG='64251fa8b5c69526337d9df77c279c5431043480fb0dbfd4faf47fa3552b08b3'


def driver():
    d=module(ORIGINAL_DRIVER,'first_win_natural_driver')
    d.OUT=OUT
    d.PARENT=d.CANDIDATE=CAND
    return d


def clearance():
    p=read(PLAN); test=p['seeds']
    nominal=[]
    for filename, section, key in RESERVED_DEVELOPMENT_PLANS:
        path=BASE/filename
        if path==PLAN:
            require(sha(path)==REG,'self-registration changed')
            continue
        content=read(path)
        nominal.extend((content if section is None else content[section])[key])
    _,known,terrains=map_identity.exposed_boards(nominal)
    registers={}
    for path in sorted((BASE/'all_top5').rglob('validation_maps.jsonl.gz')):
        rows=[json.loads(l) for l in gzip.decompress(path.read_bytes()).splitlines()]
        for row in rows:
            known.add(row['seed']);terrains.add(row['initial_terrain_u8_sha256'])
        registers[str(path.relative_to(ROOT))]={'sha256':sha(path),'games':len(rows)}
    require(any('unscored' in k for k in registers) and any('appendix_d944673f' in k and v['games']==2 for k,v in registers.items()),'missing fault / latest ladder maps')
    shards={}
    for f in [ACTOR/'data_split.json',ACTOR/'data_loss_split.json',ACTOR/'blended_actor_dev/panel.json']:
        games=read(f)['games']
        for r in games:
            known.add(r['seed']);terrains.add(r['terrain_sha256']);shards[r['id']]=r
        registers[str(f.relative_to(ROOT))]={'sha256':sha(f),'games':len(games)}
    cache_ids=[]
    for f in CACHE.glob('*.json'):
        r=read(f)
        if isinstance(r,dict) and 'file' in r and 'id' in r:
            require(r['id'] in shards,'unbound external training data')
            cache_ids.append(r['id'])
    boards={}
    for s in test:
        require(not RESERVED_RANGE[0]<=s<RESERVED_RANGE[1] and s not in known,'used seed')
        terrain=map_identity.local_board(s)
        np.random.seed(s)
        env=make('carbon',configuration={**exam.CFG,'randomSeed':s})
        require(terrain==map_identity.terrain_sha256(env.states[0].observation.full_carbon)
                and terrain not in terrains and terrain not in boards.values(),'used terrain or official opening difference')
        boards[str(s)]=terrain
    result={'registered_sha256':REG,'history_snapshot':p['source_revision'],'physical_boards':boards,
            'old_unique_seeds':len(known),'old_unique_terrains':len(terrains),'old_pledged_dev':nominal,'external_cache_ids':len(cache_ids),
            'explicit_sources':registers,'seed_collisions':0,'terrain_collisions':0}
    save(HERE/'clearance.json', result)
    return result


def admit():
    plan=read(PLAN);require(sha(PLAN)==REG and len(set(plan['seeds']))==8 and plan['seats']==[0,1] and plan['planned_games']==16,'registered schedule')
    declared=plan['candidate']['sha256'];md=driver(); original=md.arena.prior.SOURCE
    require(sha(CAND)==declared==read(ACTOR/'actor_partner_dev/release.json')['candidate_sha256'] and
            sha(original)==sha(git('show',plan['source_revision']+':'+plan['examiner']['path']))==plan['examiner']['sha256'],'strategy identity')
    files=[Path(__file__),ORIGINAL_DRIVER,ACTOR/'blended_actor_arena/clock.py',BASE/'native_clock.py',Path(md.arena.__file__),Path(md.official.__file__)]
    sources={str(f.relative_to(ROOT)):sha(f) for f in files}
    if (OUT/'binding.json').exists():
        old=read(OUT/'binding.json')
        require(old['candidate']==declared and old['checked_driver_sha256']==sources and old['registered_sha256']==REG
                and old['map_clearance_sha256']==sha(HERE/'clearance.json') and old['original']==sha(original),'resume input drift')
        return old
    audited=clearance()
    auth={'registered_sha256':REG,'candidate':declared,'original':sha(original),'boards':audited['physical_boards'],
          'checked_driver_sha256':sources,'map_clearance_sha256':sha(HERE/'clearance.json'),'role':'development','formal_games':0}
    save(OUT/'binding.json',auth)
    return auth


def one(job):
    return clock.run_one(driver(),job)


def check(job):
    arm,s,p,cpu,terrain,auth=job
    d=driver(); wall=clock.verify_completed(d,job)
    q=OUT/arm;doc=read(q/f'receipt_{s}_{p}.json');row=doc['game'];official=doc['official']
    require(d.official.check(q,row)==official and official['map_sha256']==terrain and official['statuses']==['DONE','DONE'],'terminal discrepancy')
    raw=read(q/row['raw_file']);scores=official['official_reward_by_seat']
    return dict(seed=s,seat=p,cpu=cpu,status=official['statuses'],steps=official['steps'],score_by_seat=scores,
            own=scores[p],opponent=scores[1-p],margin=scores[p]-scores[1-p],result=row['result'],extinction=raw['extinct_by_seat'],
            raw_sha256=sha(q/row['raw_file']),receipt_sha256=sha(q/f'receipt_{s}_{p}.json'),wall_sha256=sha(q/f'wall_{s}_{p}.json.gz'),
            actual_opponent_calls=len(wall['calls']))


def summarize(results):
    maps=sorted(set(r['seed'] for r in results))
    def metrics(rows):
        return {'games':len(rows),'wins':sum(r['result']=='win' for r in rows),'draws':sum(r['result']=='draw' for r in rows),
                'losses':sum(r['result']=='loss' for r in rows),'observed_win_rate':statistics.mean(r['result']=='win' for r in rows),
                'mean_official_margin':statistics.mean(r['margin'] for r in rows)}
    bymap={s:metrics([r for r in results if r['seed']==s]) for s in maps}
    rng=np.random.RandomState(5417202);index=rng.randint(0,len(maps),(50000,len(maps)))
    cis={k:np.quantile(np.array([bymap[s][k] for s in maps])[index].mean(axis=1),[.025,.975]).tolist()
         for k in ('observed_win_rate','mean_official_margin')}
    return dict(total=metrics(results),by_seat={s:metrics([r for r in results if r['seat']==s]) for s in (0,1)},by_map=bymap,
                map_bootstrap_percentile_95=cis,bootstraps=50000,games=results,formal_games=0)


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as owner, SHARED.open('a') as slot:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth=admit()
        fcntl.flock(slot,fcntl.LOCK_EX)
        results=[]
        for i,s in enumerate(read(PLAN)['seeds']):
            jobs=[('candidate',s,p,(i+p)%2,auth['boards'][str(s)],auth) for p in (0,1)]
            print('map',s,'start',flush=True)
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
                pending={pool.submit(one,j):j for j in jobs}
                for f in as_completed(pending):
                    f.result();row=check(pending[f]);results.append(row)
                    print('DONE',row['seed'],row['seat'],row['result'],row['score_by_seat'],len(results),flush=True)
                    if row['result']=='win' and not (HERE/'first_win.json').exists():
                        save(HERE/'first_win.json',dict(game=row,checked_at_unix=time.time(),signed_completed_when_reported=len(results)))
        require(len(results)==16 and len({(r['seed'],r['seat']) for r in results})==16,'incomplete')
        ordered=sorted(results,key=lambda r:(r['seed'],r['seat']))
        save(HERE/'comparison.json',summarize(ordered))

if __name__=='__main__': main()

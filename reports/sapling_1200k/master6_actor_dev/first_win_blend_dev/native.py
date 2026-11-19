"""First-win follow-up with a distinct blend-trained actor; retain old controls without replaying agents."""
from pathlib import Path
import sys
import json
import gzip
import time
import fcntl
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed

HERE=Path(__file__).resolve().parent; ACTOR=HERE.parent; OLD=ACTOR/'first_win_dev'; BASE=ACTOR.parent; ROOT=BASE.parents[1]
sys.path[:0]=[str(ACTOR/'blended_actor_arena'),str(BASE)]
from shared import sha, read, save, module, require, git
from tree_win_gate import RESERVED_RANGE
import map_identity, clock

PLAN=HERE/'protocol.json'; OUT=HERE/'games'; REG='634fe9d0b37d2c2394bca4389bef1db7afd965c1329c49a9aba0900a424c07b7'
OLD_REG='64251fa8b5c69526337d9df77c279c5431043480fb0dbfd4faf47fa3552b08b3'
source=module(OLD/'native.py','registered_first_win_sources')
follow=module(OLD/'native.py','follow_up_same_native_driver')
follow.HERE=HERE;follow.PLAN=PLAN;follow.OUT=OUT;follow.REG=REG;follow.CAND=ACTOR/'blended_actor_dev/serving/candidate.py'
LOCK=source.SHARED


def preflight():
    p=read(PLAN)
    require(sha(PLAN)==REG and sha(OLD/'protocol.json')==OLD_REG,'wrong registration')
    cc=ACTOR/'blended_actor_arena/comparison.json'
    require(sha(cc)=='a0d51b5eeb7ed1218c12fa7f45fce3de6626786d3ffb2f8e1f7fcfc33a4ac776' and
            len(read(cc)['delta_by_terrain'])==2 and min(read(cc)['delta_by_terrain'].values())>0,'candidate selection evidence')
    require(sha(follow.CAND)==p['candidate']['sha256'] and
            sha(ACTOR/'blended_actor_dev/candidate.npz')==p['candidate']['student_npz_sha256'] and
            sha(source.driver().arena.prior.SOURCE)==sha(git('show',p['source_revision']+':'+p['examiner']['path']))==p['examiner']['sha256'],'wrong frozen pair')
    if (HERE/'clearance.json').exists():
        fresh=read(HERE/'clearance.json')
        require(fresh['registered_sha256']==REG and set(map(int,fresh['physical_boards']))==set(p['seeds']),'different existing clearance')
        maps=set(fresh['physical_boards'].values());seeds=set(p['seeds'])
        for seed in seeds:
            require(fresh['physical_boards'][str(seed)]==map_identity.local_board(seed) and not RESERVED_RANGE[0]<=seed<RESERVED_RANGE[1],'stored opening differs')
        # Check for actual old or newly archived game use; the active experiment's own registration is not a historical match.
        for path in sorted((BASE/'all_top5').rglob('validation_maps.jsonl.gz')):
            for line in gzip.decompress(path.read_bytes()).splitlines():
                v=json.loads(line);require(v['seed'] not in seeds and v['initial_terrain_u8_sha256'] not in maps,'actual ladder game overlap')
        for path in (BASE/map_identity.LOCAL_EXPOSURE,ACTOR/'data_split.json',ACTOR/'data_loss_split.json',ACTOR/'blended_actor_dev/panel.json'):
            doc=read(path)
            for v in doc.get('maps',doc.get('games',[])):
                require(v['seed'] not in seeds and v.get('terrain_sha256',v.get('initial_terrain_u8_sha256')) not in maps,'new training or local map exposure')
        for path,section,key in follow.RESERVED_DEVELOPMENT_PLANS:
            if BASE/path==PLAN:continue
            doc=read(BASE/path);prior=(doc if section is None else doc[section])[key]
            require(not seeds.intersection(prior) and not maps.intersection(map_identity.local_board(seed) for seed in prior),'another registered study uses a selected map')
    else:
        fresh=follow.clearance()
    hist=read(OLD/'games/binding.json')
    require(set(p['seeds']).isdisjoint(p['historical_control_maps']) and len(set(p['seeds']))==4,'new/control overlap')
    # The 800/801 controls were fixed when the first candidate's complete 16-game schedule was declared.
    controls={}
    for s in p['historical_control_maps']:
        board=hist['boards'][str(s)]
        require(board==map_identity.local_board(s) and board not in fresh['physical_boards'].values(),'control terrain alias')
        arm=OLD/'games/candidate'
        for seat in (0,1):
            receipt=arm/f'receipt_{s}_{seat}.json'
            row=read(receipt)['game']
            require(sha(arm/row['raw_file'])==row['raw_sha256'] and read(receipt)['binding']==hist,'control tape mismatch')
            controls[f'{s}/{seat}']={'raw_sha256':row['raw_sha256'],'receipt_sha256':sha(receipt),'wall_sha256':sha(arm/f'wall_{s}_{seat}.json.gz'),'terrain':board}
    pre={'frozen':REG,'new_boards':fresh['physical_boards'],'control_source':controls,'old_protocol':OLD_REG,
         'old_binding':sha(OLD/'games/binding.json'),'old_known_maps':sha(OLD/'clearance.json'),'clearance':sha(HERE/'clearance.json')}
    if (HERE/'preflight.json').exists():
        signed=read(HERE/'preflight.json');require(signed==pre,'registered earlier source or control evidence drifted');return p,signed
    save(HERE/'preflight.json',pre);return p,pre


def admission():
    p,pre=preflight()
    old=read(OLD/'comparison.json')
    require(len(old['games'])==old['total']['games']==16,'prior incomplete')
    if old['total']['wins']:
        save(HERE/'cancelled.json',{'source_sha256':sha(OLD/'comparison.json'),'old_wins':old['total']['wins'],'new_games':0})
        return None
    require(all(r['result']!='win' and r['own']<=r['opponent'] for r in old['games']),'prior showed a win')
    files=[Path(__file__),OLD/'native.py',follow.ORIGINAL_DRIVER,ACTOR/'blended_actor_arena/clock.py',BASE/'native_clock.py',
           Path(follow.driver().arena.__file__),Path(follow.driver().official.__file__)]
    b={'registered_sha256':REG,'candidate':sha(follow.CAND),'original':p['examiner']['sha256'],'previous_full_batch':sha(OLD/'comparison.json'),
       'preflight_sha256':sha(HERE/'preflight.json'),'checked_driver_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files},'boards':dict(pre['new_boards'])}
    for s in p['historical_control_maps']:b['boards'][str(s)]=pre['control_source'][f'{s}/0']['terrain']
    if (OUT/'binding.json').exists():
        signed=read(OUT/'binding.json');require(signed==b,'registered inputs or executable changed');return signed
    save(OUT/'binding.json',b);return b


def one(job):
    return follow.one(job)


def jobs(b):
    p=read(PLAN)
    return [('candidate',s,seat,(i+seat)%2,b['boards'][str(s)],b) for group in (p['seeds'],p['historical_control_maps'])
            for i,s in enumerate(group) for seat in (0,1)]


def summary(rows):
    p=read(PLAN);fresh=[r for r in rows if r['seed'] in p['seeds']]
    controls=[r for r in rows if r['seed'] in p['historical_control_maps']]
    old=read(OLD/'comparison.json')
    require(len(fresh)==8 and len(controls)==4 and len({(r['seed'],r['seat']) for r in rows})==12,'incomplete')
    pairs=[]
    for r in controls:
        prev,=[v for v in old['games'] if (v['seed'],v['seat'])==(r['seed'],r['seat'])]
        pre=read(HERE/'preflight.json')['control_source'][f"{r['seed']}/{r['seat']}"]
        require(prev['raw_sha256']==pre['raw_sha256'] and prev['receipt_sha256']==pre['receipt_sha256'] and prev['wall_sha256']==pre['wall_sha256'],'bound historical sample')
        pairs.append({'seed':r['seed'],'seat':r['seat'],'historical':prev,'new':r,'new_minus_historical_margin':r['margin']-prev['margin']})
    return {'fresh':follow.summarize(fresh),'exposed_maps':follow.summarize(controls),'historical_pairs':pairs,
            'new_games':12,'fresh_games':8,'historical_reruns':0,'independent_formal_games':0}


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1:]==['--preflight']:
            preflight();return
        while not (OLD/'comparison.json').is_file():
            time.sleep(3)
        b=admission()
        if b is None:return
        p=read(PLAN)
        with LOCK.open('a') as court:
            fcntl.flock(court,fcntl.LOCK_EX)
            all_jobs=jobs(b);results=[]
            for first in range(0,len(all_jobs),2):
                pair=all_jobs[first:first+2]
                print('map',pair[0][1],flush=True)
                with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
                    pending={pool.submit(one,j):j for j in pair}
                    for f in as_completed(pending):
                        f.result();r=follow.check(pending[f]);results.append(r)
                        group='fresh' if r['seed'] in p['seeds'] else 'exposed'
                        print('DONE',group,r['seed'],r['seat'],r['own'],r['opponent'],r['result'],len(results),flush=True)
                        if r['result']=='win' and not (HERE/f'first_{group}_win.json').exists():
                            save(HERE/f'first_{group}_win.json',{'game':r,'official_checks_completed':len(results),'unix':time.time(),'population':group})
            save(HERE/'comparison.json',summary(sorted(results,key=lambda r:(r['seed'],r['seat']))))
if __name__=='__main__':main()

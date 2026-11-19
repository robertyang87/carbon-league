"""Run the two separately registered new maps once after published clearance and the historic screen."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
import json
import multiprocessing
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build import BASE, ROOT, CAND,PARENT,digest,out
from run import SHARED, ORIGINAL,DRIVER,require,read,native_module
sys.path.insert(0,str(BASE))
from turning_point_ledger import CashEvents,FastCarbon
from tree_win_gate import RESERVED_RANGE
from map_identity import local_board,ladder_boards

PUBLISHED='91ca86e50c17bfacab7d57d20c9571f293e321dc'
OUT=HERE/'games'


def mod():
    m=native_module();m.OUT=OUT;m.CANDIDATE=CAND;m.PARENT=PARENT
    return m


def one(job):
    return mod().one(job)


def load_json(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def admit():
    plan=read(HERE/'protocol.json');base=read(HERE.parent/'comparison.json');meta=read(HERE.parent/'release.json')
    protocol_sha=digest(HERE/'protocol.json')
    require(protocol_sha==digest(subprocess.check_output(['git','show',PUBLISHED+':'+str((HERE/'protocol.json').relative_to(ROOT))],cwd=ROOT))=='3028eb055fe5166b1e44a76c5d0a034c2afe2c5596a5e6d14f040cd62b1bded8','publication receipt mismatch')
    require(base.get('complete') and len(base['games'])==4 and len({(x['seed'],x['seat']) for x in base['games']})==4
        and sum(x['candidate_outcome']=='win' for x in base['games'])>=1 and statistics.mean([x['delta_margin'] for x in base['games']])>0
        and digest(CAND)==meta['candidate_sha256']==plan['candidate_pack_sha256'] and digest(PARENT)==plan['parent_pack_sha256'],'predeclared continuation criterion')
    prereg=plan['new_dev']; physical={}
    require(prereg['arms']==['parent','candidate'] and prereg['physical_seats']==[0,1] and len(set(prereg['seeds']))==2 and prereg['max_games']==8,'design mismatch')
    fingerprint,known_seeds,known_terrain=ladder_boards()
    additional=BASE/'all_top5'
    for p in [additional/'validation_maps.jsonl.gz',*sorted(additional.glob('appendix_*/validation_maps.jsonl.gz'))]:
        for line in gzip.decompress(p.read_bytes()).splitlines():
            row=json.loads(line);known_seeds.add(row['seed']);known_terrain.add(row['initial_terrain_u8_sha256'])
    details={}
    for s in prereg['seeds']:
        name=f'audit_{s}.json.gz';signed=HERE/name
        if signed.exists(): data=gzip.decompress(signed.read_bytes())
        else:data=(Path('/tmp')/f'partner-map-{s}.json').read_bytes()
        a=json.loads(data);t=local_board(s)
        require(a['candidate_seed']==s and a['frozen_git_revision']==plan['source_revision'] and a['current_declaration']['sha256']==protocol_sha
            and a['verdict']=='eligible_in_checked_scope' and not a['conflicts'] and not a['same_terrain_matches'] and not a['reserved']
            and not a['ladder_seed_or_terrain_match'] and a['candidate_initial_terrain_u8_sha256']==a['official_opening_sha256']==t
            and s not in known_seeds and t not in known_terrain and not RESERVED_RANGE[0]<=s<RESERVED_RANGE[1],f'unverified or conflicting candidate map {s}: {a["conflicts"]}')
        if not signed.exists(): signed.write_bytes(gzip.compress(data,mtime=0))
        physical[s]=t;details[s]={'receipt':name,'sha256':digest(signed),'counts':a['coverage']}
    require(len(set(physical.values()))==2 and digest(mod().arena.prior.SOURCE)==ORIGINAL,'board alias or examiner mismatch')
    auth=dict(protocol_sha256=protocol_sha,published_main=PUBLISHED,stage1_comparison_sha256=digest(HERE.parent/'comparison.json'),native_driver_sha256=digest(DRIVER),
        driver_sha256=digest(Path(__file__)),candidate_pack_sha256=digest(CAND),parent_pack_sha256=digest(PARENT),original_sha256=ORIGINAL,
        prior_ledger_hash=fingerprint,seen_ladder_terrains=len(known_terrain),maps=physical,clearance=details)
    out(HERE/'binding.json',auth)
    return auth,physical


def economy(row,arm):
    tape=load_json(OUT/arm/row['raw_file'])
    require(tape['pack_sha256']==(digest(CAND) if arm=='candidate' else digest(PARENT)) and tape['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1','income identity')
    sim=FastCarbon(tape['game_configuration']).reset(seed=row['seed']);opening=list(sim.cash);sim.cash=CashEvents(list(sim.cash))
    totals=[Counter(),Counter()]
    for pair in tape['actions']:
        sim.cash.events.clear();sim.step(*pair)
        for kind,s,change,*_ in sim.cash.events:totals[s][kind]+=change
    cash=list(sim.cash)
    require(all(abs(opening[s]+sum(totals[s].values())-cash[s])<.0001 for s in (0,1)),'cash accounts do not close')
    return dict(opening_cash=opening,final_cash=cash,official_score_by_seat=row['official_score_by_seat'],by_seat=[dict(c) for c in totals])


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as owner,SHARED.open('a') as slot:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(slot,fcntl.LOCK_EX)
        auth,boards=admit();pairs=[];incomes=[];arms=['parent','candidate']
        for j,(seed,board) in enumerate(boards.items()):
            for seat in (0,1):
                jobs=[(a,seed,seat,(i+seat+j)%2,board,auth) for i,a in enumerate(arms)]
                with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool: signed=list(pool.map(one,jobs))
                raw=[d['game'] for d in signed]
                for a,r in zip(arms,raw):incomes.append(dict(arm=a,seed=seed,seat=seat,raw_sha256=r['raw_sha256'],**economy(r,a)))
                pairs.append(dict(seed=seed,seat=seat,arms={a:{k:r[k] for k in ('margin','result','actual_transitions','official_score_by_seat','raw_sha256','search','wall_seconds')} for a,r in zip(arms,raw)},
                    delta_margin=raw[1]['margin']-raw[0]['margin']))
                print('complete pair',seed,seat,flush=True)
        require(len(pairs)==4,'incomplete pairing')
        bymap={s:statistics.mean(x['delta_margin'] for x in pairs if x['seed']==s) for s in boards}
        summary={a:dict(wins=sum(x['arms'][a]['result']=='win' for x in pairs),
                        wins_by_seat=[sum(x['seat']==s and x['arms'][a]['result']=='win' for x in pairs) for s in (0,1)],
                        mean_margin=statistics.mean(x['arms'][a]['margin'] for x in pairs)) for a in arms}
        out(HERE/'income.json',dict(binding_sha256=digest(HERE/'binding.json'),games=incomes))
        out(HERE/'comparison.json',dict(completed_native_games=8,pairs=pairs,arms=summary,delta_margin_by_map=bymap,
            paired_mean_margin_difference=statistics.mean(bymap.values()),map_cluster_se=statistics.stdev(bymap.values())/2**.5,independent_maps=2,
            income_sha256=digest(HERE/'income.json'),binding_sha256=digest(HERE/'binding.json'),
            interpretation='Independent of the prior historical two-map screen. The trial affords two clusters and four opportunities per policy.'))
        print('eight official games signed',digest(HERE/'comparison.json'),flush=True)


if __name__=='__main__':
    main()

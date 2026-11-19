"""Run the registered four matched pairs, resuming verified tickets under the shared native slot."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys
from shared import ACTOR,BASE,HERE,PROTO,PUBLISHED,ROOT,TRAIN,git,module,published,read,require,save,sha
from clearance import run as check_maps
from fault_exclusion import check as check_fault

BUILD='13f3f3b1253f81633f030f6b9109d207fd6ce794'
RECEIPTS='28c2653c74f057d48f5bcc8c9caf68e3c80a6aa9'
SHARED=ACTOR/'tree_response_training/.continuation.lock'
ARENA=ACTOR/'tree_tenure_dev/pay_policy/native_run.py'
OUT=HERE/'games'


def paths():
    plan=read(PROTO)
    return ROOT/plan['arm_parent']['path'],ROOT/plan['arm_candidate']['path']


def admit():
    plan=read(PROTO);old,new=paths();entry=read(HERE/'validation/entry.json.gz');block=read(HERE/'clearance.json')
    weights=TRAIN/'candidate.npz';confirmed=[weights,TRAIN/'control.npz',new,TRAIN/'serving/control.py']
    require(all(published(p,BUILD) for p in confirmed) and published(TRAIN/'serving/packages.json',RECEIPTS) and published(PROTO) and sha(old)==plan['arm_parent']['sha256'],'registered and published frozen identities')
    mods=read(TRAIN/'serving/packages.json');require(mods['preserved_source_excluding_student'] and mods['parent_quantized_roundtrip']
        and sha(new)==mods['packages']['candidate']['deployed_sha256'] and sha(weights)==mods['packages']['candidate']['npz_sha256'],'actual Sapling payload identity')
    require(entry['parent_pack_sha256']==sha(old) and entry['candidate_pack_sha256']==sha(new) and entry['protocol_sha256']==sha(PROTO) and entry['warm_calls']==2392 and entry['checked_roots']==60,'wrong actual entry')
    scores={a:read(HERE/'validation'/(a+'_ce.json')) for a in ('parent','candidate','control')}
    require(all(d['protocol_sha256']==sha(PROTO) and d['slots']==scores['parent']['slots'] and [(r['id'],r['shard_sha256'],r['slots']) for r in d['maps']]==[(r['id'],r['shard_sha256'],r['slots']) for r in scores['parent']['maps']] for d in scores.values()),'unequal CE map or valid slots')
    require(all(d['pack_sha256']==sha(pack) for d,pack in [(scores['parent'],old),(scores['candidate'],new),(scores['control'],TRAIN/'serving/control.py')]),'CE package identity')
    passed=entry['passed'] and entry['changed_roots']>=1 and scores['candidate']['equal_map_ce']<scores['parent']['equal_map_ce']
    gate=dict(passed=passed,entry_changed=entry['changed_roots'],held_maps=len(scores['parent']['maps']),common_slots=scores['parent']['slots'],
              ce={k:d['equal_map_ce'] for k,d in scores.items()},published_protocol=PUBLISHED,published_weights=BUILD,published_pack_receipt=RECEIPTS,
              protocol_sha256=sha(PROTO),entry_sha256=sha(HERE/'validation/entry.json.gz'),
              ce_artifacts={arm:sha(HERE/'validation'/(arm+'_ce.json')) for arm in scores},new_development_boards_only=block['physical_boards'])
    save(HERE/'admission.json',gate)
    require(passed,'registered native admission did not pass, no original match')
    if (OUT/'binding.json').exists():
        a=read(OUT/'binding.json');require(a['gate_sha256']==sha(HERE/'admission.json') and a['parent']==sha(old) and a['candidate']==sha(new),'resumption identity drift')
        return a
    # Scan the pre-game public tree once. All later resumption continues these exact tickets.
    git('fetch','-q','origin','main');current=git('rev-parse','origin/main').decode().strip()
    check=check_maps(current,publish=False)
    check_fault()
    require(check['eligible_as_development_only'] and check['physical_boards']=={int(s):t for s,t in block['physical_boards'].items()},'incremental exposure')
    save(HERE/'clearance_tip.json',check)
    m=driver()
    require(sha(m.arena.prior.SOURCE)==plan['native']['original_sha256']==sha(git('show',BUILD+':'+plan['native']['original'])),'original examiner')
    own=[Path(__file__),HERE/'clock.py',ARENA,Path(m.arena.__file__),BASE/'native_clock.py']
    auth={'gate_sha256':sha(HERE/'admission.json'),'protocol_sha256':sha(PROTO),'build_revision':BUILD,
          'clearance_pre_sha256':sha(HERE/'clearance.json'),'fault_exposure_sha256':sha(HERE/'clearance_fault.json'),'clearance_tip_sha256':sha(HERE/'clearance_tip.json'),
          'parent':sha(old),'candidate':sha(new),'original_sha256':sha(m.arena.prior.SOURCE),
          'checked_driver_sha256':{str(p.relative_to(ROOT)):sha(p) for p in own},'boards':block['physical_boards']}
    save(OUT/'binding.json',auth);return auth


def driver():
    m=module(ARENA,'blend_frozen_original_driver');m.OUT=OUT;m.PARENT,m.CANDIDATE=paths();return m


def one(job):
    import clock
    return clock.run_one(driver(),job)


def collect(auth):
    template=ACTOR/'actor_partner_dev/replication/native.py'
    sys.path.insert(0,str(template.parent.parent))
    prior=module(template,'partner_economy_reuse');prior.PARENT,prior.CAND=paths();prior.OUT=OUT
    pairs=[];econ=[];tot={arm:[] for arm in ('parent','candidate')}
    import clock as audit_clock
    for map_index,(s,tmap) in enumerate(auth['boards'].items()):
        seed=int(s)
        for seat in (0,1):
            row={'seed':seed,'seat':seat,'arms':{}}
            for arm in tot:
                job=(arm,seed,seat,((arm=='candidate')+map_index+seat)%2,tmap,auth)
                audit_clock.verify_completed(driver(),job)
                signed=read(OUT/arm/f'receipt_{seed}_{seat}.json');r=signed['game']
                require(signed['binding']==auth and signed['official']['map_sha256']==tmap and r['margin']==signed['official']['official_reward_by_seat'][seat]-signed['official']['official_reward_by_seat'][1-seat],'raw/official mismatch')
                stats=prior.economy(r,arm);raw=read(OUT/arm/r['raw_file']);clock=read(OUT/arm/f'wall_{seed}_{seat}.json.gz')
                require(clock['raw_sha256']==r['raw_sha256'] and len(clock['calls'])==r['actual_transitions'],'timing coverage')
                econ.append(dict(seed=seed,seat=seat,arm=arm,raw_sha256=r['raw_sha256'],**stats))
                row['arms'][arm]={'win':r['result']=='win','result':r['result'],'official_margin':r['margin'],'score_by_seat':raw['official_score_by_seat'],
                   'steps':raw['steps'],'terminal_extinction':raw['extinct_by_seat'],'raw_sha256':r['raw_sha256'],'wall_sha256':sha(OUT/arm/f'wall_{seed}_{seat}.json.gz')}
                tot[arm].append((seat,row['arms'][arm]))
            row['delta_official_margin']=row['arms']['candidate']['official_margin']-row['arms']['parent']['official_margin'];pairs.append(row)
    require(len(pairs)==4,'all four matched seats')
    terrain={s:statistics.mean(r['delta_official_margin'] for r in pairs if r['seed']==int(s)) for s in auth['boards']}
    summary={arm:dict(wins=sum(r['win'] for _,r in vals),draws=sum(r['result']=='draw' for _,r in vals),
                      mean_official_margin=statistics.mean(r['official_margin'] for _,r in vals),
                      seats={s:dict(wins=sum(r['win'] for seat,r in vals if seat==s),mean_official_margin=statistics.mean(r['official_margin'] for seat,r in vals if seat==s)) for s in (0,1)}) for arm,vals in tot.items()}
    save(HERE/'income.json',econ)
    save(HERE/'comparison.json',dict(completed_games=8,gate_sha256=auth['gate_sha256'],pairs=pairs,by_arm=summary,delta_by_terrain=terrain,
                    paired_delta=statistics.mean(terrain.values()),cluster_se=statistics.stdev(terrain.values())/(2**.5),
                    income_sha256=sha(HERE/'income.json'),independent_formal_games=0))


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);auth=admit()
        for i,(s,board) in enumerate(auth['boards'].items()):
            for seat in (0,1):
                jobs=[(a,int(s),seat,(j+i+seat)%2,board,auth) for j,a in enumerate(('parent','candidate'))]
                if all((OUT/a/f'receipt_{s}_{seat}.json').exists() and (OUT/a/f'wall_{s}_{seat}.json.gz').exists() for a,*_ in jobs):continue
                print('waiting for shared native CPUs',s,seat,flush=True)
                with SHARED.open('a') as slot:
                    fcntl.flock(slot,fcntl.LOCK_EX)
                    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
                        for r in pool.map(one,jobs):print(r['official'],flush=True)
        collect(auth)


if __name__=='__main__':main()

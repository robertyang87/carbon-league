"""Run only the registered four development games with the original examiner and check official rewards."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
from build import BASE, CAND, FROZEN, HERE, PARENT, ROOT, build, digest, out

SHARED=HERE.parent/'tree_response_training/.continuation.lock'
DRIVER=HERE.parent/'tree_tenure_dev/pay_policy/native_run.py'
NATIVE=HERE/'native'
ORIGINAL='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'


def read(path):
    return json.loads(path.read_bytes())


def require(test, label):
    if not test:raise ValueError(label)


def native_module():
    sp=importlib.util.spec_from_file_location('partner_shared_native',DRIVER)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.OUT=NATIVE;m.CANDIDATE=CAND
    return m


def one(job):
    return native_module().one(job)


def setup():
    release=build();m=native_module()
    require(digest(m.arena.prior.SOURCE)==ORIGINAL==digest(subprocess.check_output(['git','show',FROZEN+':sources/master3s2/master3s2.py'],cwd=ROOT)), 'original identity')
    probe=read(HERE/'mechanism.json');plan=read(HERE/'protocol.json')
    require(probe['passed'] and probe['candidate_sha256']==digest(CAND) and probe['protocol_sha256']==digest(HERE/'protocol.json'),'mechanism failed')
    parents={};inputs={}
    for seed in plan['maps']:
        folder=HERE.parent/plan['parent_source'][str(seed)]
        report=read(folder/'results.json')
        require(report['status']=='complete' and report['header']['pack_sha256']==digest(PARENT),'wrong historical parent')
        stored=read(folder/'official_replay.json')['runs'][folder.name]['games']
        for path in [folder/'results.json',folder/'official_replay.json']:
            inputs[str(path.relative_to(ROOT))]=digest(path)
        for row in report['games']:
            if row['seed']!=seed: continue
            audit=m.official.check(folder,row)
            require(audit in stored and row['valid'] and not row['errors'] and row['search']['searched']>0,'bad historical native game')
            key=(seed,row['candidate_seat']);require(key not in parents,'duplicate historical seat')
            parents[key]=row
    require(set(parents)=={(s,k) for s in plan['maps'] for k in (0,1)},'unpaired original')
    binding=dict(release_sha256=digest(HERE/'release.json'),protocol_sha256=digest(HERE/'protocol.json'),
                 historical_sources=inputs,native_step_driver_sha256=digest(DRIVER),runner_sha256=digest(HERE/'run.py'),
                 original_sha256=ORIGINAL,mechanism_sha256=digest(HERE/'mechanism.json'),candidate_sha256=digest(CAND),
                 sapling_npz_sha256=release['input_sha256']['data/weights/sapling_1200k/master6_actor_v1.npz'])
    return plan,binding,parents


def main():
    NATIVE.mkdir(exist_ok=True)
    with (NATIVE/'.owner.lock').open('a') as owner, SHARED.open('a') as shared:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX)
        plan,binding,parents=setup();out(NATIVE/'binding.json',binding)
        records=[]
        for seed in plan['maps']:
            work=[('candidate',seed,s,s,parents[(seed,s)]['initial_terrain_u8_sha256'],binding) for s in plan['seats']]
            with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
                for item in pool.map(one,work):
                    r=item['game'];p=parents[(r['seed'],r['candidate_seat'])]
                    records.append(dict(seed=seed,seat=r['candidate_seat'],candidate=r['official_score_by_seat'],parent=p['official_score_by_seat'],
                        candidate_margin=r['margin'],parent_margin=p['margin'],delta_margin=r['margin']-p['margin'],
                        candidate_outcome=r['result'],parent_outcome=p['result'],candidate_sha256=r['raw_sha256'],parent_sha256=p['raw_sha256'],
                        candidate_search=r.get('search'),parent_search=p['search'],candidate_t=r['actual_transitions'],parent_t=p['actual_transitions']))
            print('signed both seats',seed,flush=True)
        scores=[statistics.mean(a['delta_margin'] for a in records if a['seed']==s) for s in plan['maps']]
        out(HERE/'comparison.json',dict(binding=binding,development_only=True,complete=True,games=records,per_map_delta=scores,
            candidate_wins=sum(x['candidate_outcome']=='win' for x in records),parent_wins=sum(x['parent_outcome']=='win' for x in records),
            wins_by_seat=[sum(x['seat']==s and x['candidate_outcome']=='win' for x in records) for s in (0,1)],
            mean_margin=statistics.mean(x['candidate_margin'] for x in records),mean_paired_change=statistics.mean(scores),
            map_cluster_se=statistics.stdev(scores)/len(scores)**.5,independent_map_clusters=len(scores),
            limitation='Two exposed development maps; historical parent and new candidate had independent original search time budgets and unseeded draws.'))
        print('completed',digest(HERE/'comparison.json'),flush=True)


if __name__=='__main__':
    main()

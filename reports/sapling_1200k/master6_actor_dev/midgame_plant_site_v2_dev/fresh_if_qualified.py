"""Advance to one distinct fresh paired development map only after the two real-branch gates."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import subprocess
import sys
from package_sites_v2 import build,verify
from study import HERE,ACTOR,PACK,sr,plan

sys.path.insert(0,str(ACTOR/'first_win_dev'))
import native

CAND=HERE/'candidate.py';PARENT=PACK/'candidate.py';SEED=621758337
def play(job):
    driver=native.driver();driver.CANDIDATE=CAND;driver.PARENT=PARENT
    native.clock.run_one(driver,job)
    checked=native.check(job);arm=job[0]
    raw=sr.load(native.OUT/arm/f'raw_{SEED}_{job[2]}.json.gz')
    if raw['pack_sha256']!=sr.sha(CAND if arm=='candidate' else PARENT):
        raise ValueError('raw pack identity drift')
    return checked

def run():
    result=sr.load(HERE/'native_site_settlement.json')
    if not result['eligible_for_fresh_map']:raise ValueError('both-site advancement gate failed')
    build(CAND);verify(CAND)
    final=sr.load(HERE/'site_package_parity.json')
    if len(final['matches'])!=2 or sr.sha(CAND)!=final['sha256'] or sr.sha(PARENT)!=plan['source_hash']:
        raise ValueError('candidate/parent parity failed')
    native.HERE=HERE;native.PLAN=HERE/'fresh_registration.json';native.OUT=HERE/'games'
    original=sr.sha(sr.ROOT/'data/master3s2_source.py')
    if original!='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5':raise ValueError('examiner changed')
    registered=dict(seeds=[SEED],seats=[0,1],arms=['parent','candidate'],formal=False,
                    original=original,parent=sr.sha(PARENT),candidate=sr.sha(CAND),
                    branch_settlement=sr.sha(HERE/'native_site_settlement.json'),policy_protocol=sr.sha(HERE/'site_protocol.json'))
    if native.PLAN.exists():
        old=sr.load(native.PLAN);registered['source_revision']=old['source_revision']
        if old!=registered:raise ValueError('fresh ticket changed')
    else:
        registered['source_revision']=subprocess.check_output(['git','-C',str(sr.ROOT),'rev-parse','HEAD'],text=True).strip()
        sr.retain(native.PLAN,registered)
    native.REG=sr.sha(native.PLAN)
    with native.SHARED.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        terrain=native.clearance()['physical_boards'][str(SEED)]
        # Also examine later nearby development files beyond the normal seed registries.
        references=set()
        for raw in (ACTOR).rglob('raw_*.json.gz'):
            if HERE in raw.parents:continue
            references.add(raw.resolve())
        for raw in references:
            doc=sr.load(raw)
            if doc.get('seed')==SEED or doc.get('initial_terrain_u8_sha256')==terrain:
                raise ValueError('target map exists in prior development evidence')
        sr.retain(HERE/'additional_fresh_check.json',dict(development_raw_tapes_checked=len(references),terrain_sha256=terrain))
        bind=dict(protocol=native.REG,candidate=sr.sha(CAND),parent=sr.sha(PARENT),original=original,terrain=terrain)
        rows=[]
        for arm in ('parent','candidate'):
            jobs=[(arm,SEED,s,s if arm=='parent' else 1-s,terrain,bind) for s in (0,1)]
            with ProcessPoolExecutor(2) as workers:
                rows.extend(dict(arm=arm,**r) for r in workers.map(play,jobs))
        sr.retain(HERE/'scores.json',rows)
if __name__=='__main__':run()

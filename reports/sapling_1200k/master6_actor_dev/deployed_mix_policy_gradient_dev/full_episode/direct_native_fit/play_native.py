"""Execute exactly eight exposed, complete original-Master3s2 training games with alternating paired CPU assignments."""
import fcntl,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from register_fit import ROOT,HERE,ACTOR,sha
sys.path.insert(0,str(ACTOR/'first_win_dev'));import native
native.HERE=HERE; native.OUT=HERE/'games';native.PLAN=HERE/'protocol.json';native.REG=sha(native.PLAN)
plan=native.read(native.PLAN);base_driver=native.driver

def bound():
    d=base_driver();d.CANDIDATE=HERE/'plus.py';d.PARENT=HERE/'minus.py';b=native.read(HERE/'binding.json')
    assert sha(d.CANDIDATE)==b['signs']['plus']['pack'] and sha(d.PARENT)==b['signs']['minus']['pack']
    assert sha(d.arena.prior.SOURCE)==plan['examiner']['sha256']
    return d
native.driver=bound
def one(job):
    arm,seed,seat,cpu,terrain,auth=job
    native.clock.run_one(bound(),job)
    z=native.check(job)
    sign='plus' if arm=='candidate' else 'minus';z.update(sign=sign,driver_label=arm,pack=auth[sign])
    native.save(HERE/f'complete_{sign}_{seed}_{seat}.json',z)
    return z
def main():
    b=native.read(HERE/'binding.json');assert b['protocol']==native.REG
    if not b['signs']:return
    maps=native.read(HERE/'maps.json')['boards'];d=bound()
    auth=dict(protocol=native.REG,source=sha(Path(__file__)),map_audit=sha(HERE/'maps.json'),calibration=sha(HERE/'binding.json'),
              center=plan['center']['sha256'],original=plan['examiner']['sha256'],plus=sha(d.CANDIDATE),minus=sha(d.PARENT),
              driver=sha(native.ORIGINAL_DRIVER),clock=sha(Path(native.clock.__file__)))
    native.OUT.mkdir(exist_ok=True)
    allrows=[]
    with (HERE/'.own.lock').open('a') as own,native.SHARED.open('a') as clk:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(clk,fcntl.LOCK_EX)
        for i,wave in enumerate(plan['waves']):
            jobs=[('candidate' if sign=='plus' else 'parent',seed,seat,cpu,maps[str(seed)],auth) for sign,seed,seat,cpu in wave]
            with ProcessPoolExecutor(2) as pool:
                for fut in as_completed([pool.submit(one,j) for j in jobs]):
                    z=fut.result();allrows.append(z)
                    print('DONE',len(allrows),z['sign'],z['seed'],z['seat'],z['own'],z['opponent'],z['margin'],z['result'],flush=True)
            native.save(HERE/f'wave_{i+1}.json',sorted(allrows,key=lambda r:(r['seed'],r['seat'],r['sign'])))
    assert len({(r['sign'],r['seed'],r['seat']) for r in allrows})==len(allrows)==8
    native.save(HERE/'scores.json',sorted(allrows,key=lambda r:(r['seed'],r['seat'],r['sign'])))
if __name__=='__main__':main()

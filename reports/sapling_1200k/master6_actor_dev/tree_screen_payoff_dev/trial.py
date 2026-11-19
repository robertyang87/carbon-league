"""Run the first accepted root on each distinct map under the shared examiner clock lock."""
from fit import HERE,ACTOR,sr
import fcntl,subprocess,sys
sys.path.insert(0,str(ACTOR/'capture_pressure_branches'))
import compare as runner
runner.HERE=HERE;runner.PRIOR=ACTOR/'planter_capture_full_dev'
def selections():
    plan=sr.load(HERE/'protocol.json');seen=set();ls=[]
    for r in sr.load(HERE/'probe.json'):
        if r['pack']==plan['parent'] and r['score']['accept'] and r['seed'] not in seen:
            ls.append(r);seen.add(r['seed'])
    assert [(r['seat'],r['t']) for r in ls]==[(1,39),(0,193)]
    return ls
def prepare():
    for r in selections():
        path=sr.ROOT/r['source'];raw=sr.load(path);case=r['root'];s=r['seat'];out=HERE/f'seat{s}';out.mkdir(exist_ok=True)
        doc=sr.load(path.with_name(path.name.replace('raw_','receipt_').replace('.json.gz','.json')))
        assert sr.sha(path)==doc['game']['raw_sha256'] and raw['pack_sha256']==sr.sha(runner.PRIOR/'candidate.py')
        sr.retain(out/'official_replay.json',{'runs':{out.name:{'games':[doc['official']]}}})
        sr.retain(out/'live_protocol.json',dict(seed=r['seed'],seat=s,root_t=r['t'],family='tree_screen_payoff',
            source_dir=str(out.relative_to(sr.ROOT)),pack_sha256=raw['pack_sha256'],
            record=dict(seed=r['seed'],seat=s,source=r['source'],source_sha256=sr.sha(path)),
            root=dict(t=r['t'],worker=case['uid'],target=case['enemy'],tree=case['tree'],old=case['before'] or 'NONE',alternative=case['after'])))
        sr.live.prepare(out)
if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:runner.play(int(sys.argv[1]),sys.argv[2])
    else:
        for s in (1,0):
            with (runner.TREE/'.continuation.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                jobs=[subprocess.Popen(['taskset','-c',str((s+i)%2),sys.executable,__file__,str(s),arm]) for i,arm in enumerate(('parent','approach'))]
                if any([j.wait() for j in jobs]):raise RuntimeError('continuation failed')

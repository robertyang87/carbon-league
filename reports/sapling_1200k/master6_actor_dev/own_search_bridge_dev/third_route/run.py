import copy
import fcntl
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
TREE=HERE.parents[1]/'tree_response_training'
sys.path.insert(0,str(TREE))
import select_roots as sr

def choose():
    cases=[]
    for batch in [TREE,*sorted((TREE/'batches').glob('intake_*')),*sorted((TREE/'supplement_01/batches').glob('intake_*'))]:
        cases.extend(sorted(batch.glob('case_*/live_protocol.json')))
    assert len(cases)==64
    plans=[(p,sr.load(p)) for p in cases]
    selected=[];seen=set()
    for seat in (0,1):
        for path,q in sorted(plans,key=lambda v:(-v[1]['root_t'],v[1]['seed'],str(v[0]))):
            if q['seat']!=seat or q['seed'] in seen:continue
            root=sr.load(path.parent/'root.json.gz')
            if root['old']!='NONE':continue
            item=copy.deepcopy(root['candidate'])
            item['actions'][root['new']]['friendly_conflict']=True
            replica=sr.route(sr.load(sr.ROOT/q['record']['source']),item)
            if replica is None:continue
            assert replica['v1']==root['v1'] and replica['new'] not in (root['old'],root['new'])
            out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
            q=dict(q,family='third_planter_route',root={**q['root'],'alternative':replica['new']},
                   old_protocol_sha256=sr.sha(path),selection_sha256=sr.sha(HERE/'protocol.md'))
            sr.retain(out/'live_protocol.json',q)
            sr.retain(out/'live_shared_init.json',sr.load(path.parent/'live_shared_init.json'))
            selected.append(dict(seat=seat,seed=q['seed'],step=q['root_t'],old_route=root['new'],new_route=replica['new'],source=str(path.parent.relative_to(sr.ROOT))))
            seen.add(q['seed']);break
    assert len(selected)==2
    sr.retain(HERE/'selection.json',selected)

def one(seat,arm):
    out=HERE/f'seat{seat}'
    sr.live.PACK=sr.SOURCE;sr.live.PACK_HASH=sr.PLAN['parent_pack_sha256']
    sr.live.run(arm,out,sr.load(out/'live_protocol.json'),resume=True)

if __name__=='__main__':
    if len(sys.argv)>1:
        one(int(sys.argv[1]),sys.argv[2])
    else:
        with (TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            choose()
            for seat in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(('parent','third'))]
                status=[j.wait() for j in jobs]
                if any(status):raise RuntimeError(status)

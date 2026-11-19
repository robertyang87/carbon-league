"""Inspect the largest observed U spread after the frozen update, using only the supplied historical public observations."""
from pathlib import Path
from run import HERE, ROOT, read, write, sha, policy
from world import World


def main():
    files=sorted((HERE/'training').glob('*.json.gz'))
    rows=[read(p) for p in files]
    def spread(d):
        r=d['root'];u=[b['official']['u_by_seat'][r['seat']] for b in d['outcomes'].values()]
        return max(u)-min(u)
    d=max(rows,key=spread);r=d['root'];s=r['seat'];t=r['t'];ref=read(ROOT/r['source'])
    greedy=__import__('json').loads(r['greedy'])
    edits=sorted({uid for k in d['outcomes'] for uid in __import__('json').loads(k).keys()|greedy.keys()
                  if __import__('json').loads(k).get(uid)!=greedy.get(uid)})
    def frame(w):
        o=w.obs(s)
        p=o['players'][s];en=o['players'][1-s]
        def item(uid):
            worker=p[2].get(uid)
            if not worker:return None
            pos,load,role=worker
            def dist(cell):
                dy=abs(cell//15-pos//15);dx=abs(cell%15-pos%15)
                return min(dx,15-dx)+min(dy,15-dy)
            near=[dict(uid=k,cell=v[0],load=v[1],role=v[2]) for k,v in en[2].items() if dist(v[0])<=2]
            return dict(cell=pos,load=load,role=role,nearby_enemies=near)
        return dict(t=w.step,cash_by_seat=[x[0] for x in o['players']],workers={uid:item(uid) for uid in edits})
    evidence=[]
    for k,b in d['outcomes'].items():
        w=World(r['seed']);selected=set([t-10,t-5,t-3,t,t+1,t+3,t+5,t+10]);f=[]
        for j,commands in enumerate(ref['actions'][:t]+b['actions']):
            if j in selected:
                z=frame(w);z['actual_command']={uid:commands[s].get(uid,'NONE') for uid in edits};f.append(z)
            w.advance(commands)
        evidence.append(dict(selected_command=k,greedy=k==r['greedy'],draws=r['draws'].count(k),frames=f,official=b['official']))
    write(HERE/'turning_example.json',dict(seed=r['seed'],seat=s,t=t,role_edited=edits,practice_opponent=r['opponent'],branches=evidence,
          source_sha256=r['source_sha256'],selection='post-hoc largest root U range in the existing training set; common observed history until t and exact executed continuations thereafter'))
if __name__=='__main__':main()

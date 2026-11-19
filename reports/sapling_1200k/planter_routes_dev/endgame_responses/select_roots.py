"""Freeze geometry-selected late roots before evaluating any intervention."""
import gzip,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/'root_recovery_branches'))
from branches import MOVES,destination,public,begin,distances,clean
B=HERE.parent/'candidate'; tape=B/'raw_621456920_1.json.gz'
def read(p):return json.loads(p.read_bytes())
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def select():
    plan=read(HERE/'protocol.json');raw=json.loads(gzip.decompress(tape.read_bytes()));seat=plan['seat'];seed=plan['map_seed'];n=raw['game_configuration']['size']
    assert raw['pack_sha256']==plan['candidate_sha256'] and (raw['seed'],raw['candidate_seat'])==(seed,seat)
    d=distances(n);env=begin(raw);T=raw['game_configuration']['episodeSteps']-1
    hits={'deliver':[],'deny':[]}
    for t,joint in enumerate(raw['actions'][:plan['roots'][1]+1]):
        if t<plan['roots'][0]:env.step(joint);continue
        o=public(env,seat);p,e=o['players'][seat],o['players'][1-seat];ours,theirs=p[2],e[2];our_base=list(p[1].values());enemy_base=list(e[1].values())
        a=joint[seat];dest={u:destination(w[0],a.get(u) or 'NONE',n) for u,w in ours.items()}
        recruits={cell for uid,cell in p[1].items() if a.get(uid) in ('RECCOLLECTOR','RECPLANTER')}
        foe_trees={w[0] for w in e[3].values()};left=T-t
        for uid,(pos,load,typ) in ours.items():
            if typ!='COLLECTOR':continue
            old=a.get(uid) or 'NONE';banned={x for u,x in dest.items() if u!=uid}|set(enemy_base)|recruits
            h=min(d[pos][h] for h in our_base)
            for i,act in enumerate(MOVES):
                z=destination(pos,act,n)
                if act==old or z in banned:continue
                threat=lambda ignore:any(v[2]=='COLLECTOR' and v[1]<=load and d[z][v[0]]<=1 for u,v in theirs.items() if u!=ignore)
                common=dict(t=t,seat=seat,worker=uid,origin=pos,load=round(load,3),old=old,replacement=act,target_cell=z,root_cash=[p[0],e[0]],remain=left,own_base=our_base,rival_base=enemy_base)
                if load>=20 and 0<h<=left and min(d[dest[uid]][b] for b in our_base)>=h and min(d[z][b] for b in our_base)<h and not threat(None):
                    hits['deliver'].append(([-load,left-h,t,uid,int(z in foe_trees),i],dict(**common,home_distance=h,on_hostile_tree=z in foe_trees)))
                for eid,(ep,ec,role) in theirs.items():
                    if role!='COLLECTOR' or ec<40 or load>=ec or threat(eid):continue
                    eh=min(d[ep][b] for b in enemy_base)
                    if not 2<=eh<=min(5,left) or d[ep][z]!=1 or min(d[z][b] for b in enemy_base)!=eh-1:continue
                    short=[destination(ep,k,n) for k in list(MOVES)[1:] if min(d[destination(ep,k,n)][b] for b in enemy_base)==eh-1]
                    hits['deny'].append(([-ec,len(short),t,load,uid,i,eid],dict(**common,enemy_worker=eid,enemy_origin=ep,enemy_cargo=round(ec,3),enemy_home_distance=eh,enemy_shortest_destinations=short)))
        env.step(joint)
    selected={}
    for kind in ('deliver','deny'):
        cand=sorted(hits[kind],key=lambda row:row[0])
        for _,row in cand:
            if any((row['t'],row['worker'])==(r['t'],r['worker']) for r in selected.values()):continue
            selected[kind]=row;break
    out=dict(selection_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(tape.read_bytes()).hexdigest(),all_publicly_eligible_counts={k:len(v) for k,v in hits.items()},all_public_matches={k:[x[1] for x in sorted(v,key=lambda r:r[0])] for k,v in hits.items()},selected=selected)
    write(HERE/'roots.json',out)
    print(json.dumps(selected,ensure_ascii=False))
if __name__=='__main__':select()

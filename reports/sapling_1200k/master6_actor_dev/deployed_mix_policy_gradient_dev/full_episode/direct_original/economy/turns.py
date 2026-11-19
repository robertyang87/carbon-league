"""For each predefined cash failure metric retain the worst ten consecutive turns and both physical action tapes."""
from collections import Counter
from opportunities import *
def track_state(states,orders,t,s,anchor_uid,target,anchor_t):
    shared=states[t][s];players=shared['players']
    def worker(owner,uid):
        v=players[owner][2].get(uid)
        if not v:return None
        p,c,r=v;order=orders[t][owner].get(uid) or 'STAY';dest=int(TO[p,DIRECTIONS.index(order)])
        return dict(id=uid,pos=p,role=r,cargo=c,order=order,next_pos=dest,distance_to_site=int(D[p,target]),
                    nearest_own_base=min(int(D[p,k]) for k in players[owner][1].values()))
    def terrain(owner):
        visible=set()
        for p,_,r in players[owner][2].values():
            if r=='COLLECTOR':visible.add(p)
            if r=='PLANTER':visible.update(map(int,TO[p,1:]))
        return dict(known=sum(k in visible for k in TO[target,1:]),visible_neighbor_carbon=sum(states[t][owner]['carbon'][int(k)] for k in TO[target,1:] if k in visible))
    teams=[]
    for who in (s,1-s):
        best=[]
        for role in ('COLLECTOR','PLANTER'):
            ids=sorted([k for k,w in players[who][2].items() if w[2]==role],key=lambda k:(int(D[players[who][2][k][0],target]),k))
            if ids:best.append(worker(who,ids[0]))
        teams.append(dict(cash=players[who][0],trees=len(players[who][3]),visible_site=terrain(who),nearest_by_role=best,full_joint_action=orders[t][who]))
    anchor=worker(1-s,anchor_uid) if anchor_uid else None
    lighter=[]
    if anchor and anchor['role']=='COLLECTOR':
        qualified=sorted([k for k,w in players[s][2].items() if w[2]=='COLLECTOR' and w[1]<anchor['cargo']],key=lambda k:(int(D[players[s][2][k][0],anchor['pos']]),k))
        if qualified:
            v=worker(s,qualified[0]);v['distance_to_carrier']=int(D[v['pos'],anchor['pos']]);lighter.append(v)
    owner=next((k for k,pl in enumerate(players) if any(pos==target for pos,age in pl[3].values())),None)
    return dict(t=t,relative=t-anchor_t,tracked_enemy=anchor,closest_own_lighter_to_enemy_carrier=lighter,site_owner=owner,
                own=teams[0],opponent=teams[1])
def run():
    compiled=[]
    for c in index():
        if c['source']!='direct343':continue
        _,steps,ev,states,orders=process(c,True);s=c['seat'];opp=1-s
        for kind,lb,ub in [('own_delivery',200,299),('tree_ground',100,200)]:
            def amount(t):
                acc=steps[t]['accounts'];return acc[opp].get(kind,0)-acc[s].get(kind,0)
            start=max(range(lb,ub-9),key=lambda t:sum(amount(u) for u in range(t,t+10)))
            delta=sum(amount(u) for u in range(start,start+10))
            if kind=='own_delivery':
                cand=[x for x in ev if x['kind']==kind and x['recipient']==opp and start<=x['t']<start+10]
                focal=max(cand,key=lambda x:x['value']);uid,cell=focal['ident'];anchor=focal['t'];detail=dict(event='normal_bank',uid=uid,cell=cell,fee_or_receipt=focal['value'])
            else:
                tracks={}
                for x in ev:
                    if x['kind']=='tree_absorption' and x['recipient']==opp and start<=x['t']<start+10:
                        tid,cell,tax=x['ident'];v=tracks.setdefault(tid,dict(pos=cell,ground=0.));v['ground']+=x['value']-tax
                tid=max(tracks,key=lambda k:tracks[k]['ground']);cell=tracks[tid]['pos']
                def occupied(t):
                    if t==len(states):return False
                    return tid in states[t][s]['players'][opp][3]
                epoch=next(t for t in range(start,start+11) if occupied(t))
                while epoch>0 and occupied(epoch-1):epoch-=1
                tx=[x for x in ev if x['recipient']==opp and x['t']==epoch-1 and x['kind'] in ('plant','seize') and x['ident'][1]==cell]
                if len(tx)!=1:raise ValueError(('unbound tree',tid,epoch,cell))
                uid=tx[0]['ident'][0];anchor=epoch-1
                detail=dict(event=tx[0]['kind'],uid=uid,tree_id=tid,cell=cell,fee_or_receipt=-tx[0]['value'],top_tree_window_ground=tracks[tid]['ground'])
            historical=[track_state(states,orders,max(0,anchor-j),s,uid,cell,anchor) for j in (10,5,3,0)]
            compiled.append(dict(arm=c['arm'],seat=s,source=c['sha256'],category=kind,window=[start,start+9],opponent_net_advantage=delta,
                  root=anchor,tracked=detail,before=historical,criterion='Max opponent-minus-own linked account over a ten-step moving window within the named phase, earliest tie. For banking select the largest opponent deposit; for soil select the tree contributing the most soil income, look back from the purchase/takeover. Retrospective selection.'))
    write('four_games_turning_points.json',compiled)
    for r in compiled:
        if r['arm']=='candidate':
            print(r['seat'],r['category'],r['window'],round(r['opponent_net_advantage']),r['root'],r['tracked'])
            for h in r['before']:
                a=h['tracked_enemy'];ca=h['closest_own_lighter_to_enemy_carrier'];print(h['t'],[a['pos'],a['order'],round(a['cargo'])] if a else None,ca[0]['distance_to_carrier'] if ca else None,[(v['role'],v['pos'],v['order']) for v in h['own']['nearest_by_role']])
if __name__=='__main__':run()

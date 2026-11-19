"""Trace each actual enemy freight toll to its tree, payer, action and prior public route."""
from collections import defaultdict
import sys
from measure import HERE,sr,_D
from turning_point_ledger import SITES


class TollAccount(list):
    def __init__(self, cash):
        super().__init__(cash)
        self.tolls=[]

    def __setitem__(self, seat, value):
        frame=sys._getframe(1)
        if frame.f_code is not sr.FastCarbon.step.__code__:raise RuntimeError("unexpected cash writer")
        if SITES[frame.f_lineno]=='tree_absorption':
            f=frame.f_locals;tree=f['t'];sim=f['self'];total=0.
            for p in [tree.pos,*sim._nbr8[tree.pos]]:
                uid=f['cell_worker'].get(p)
                if uid is None:continue
                w=sim.workers[uid]
                if w.is_collector and w.player!=seat and w.carbon>0:
                    part=w.carbon*f['coll_rate'];total+=part
                    self.tolls.append((tree.id,tree.pos,w.player,uid,w.pos,part))
            if total>value-self[seat]+1.e-7:raise AssertionError("attributed toll exceeds actual transfer")
        super().__setitem__(seat,value)


def run():
    rows=[];summary=[]
    games=sr.load(HERE.parent/'tactical_exchange_dev/paired.json.gz')
    for g in games:
        raw=sr.load(sr.ROOT/g['path']);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=g['seed'])
        sim.cash=TollAccount(sim.cash);past=[];s=g['seat'];counts=[defaultdict(float),defaultdict(float)]
        for t,joint in enumerate(raw['actions']):
            obs=sim.obs(s);past.append([{k:[v[0],v[1],joint[a].get(k)] for k,v in side[2].items() if v[2]=='COLLECTOR'} for a,side in enumerate(obs['players'])])
            sim.cash.tolls.clear();sim.step(*joint)
            for tree,cell,side,uid,z,v in sim.cash.tolls:
                pre=past[t][side].get(uid)
                if pre is None:raise AssertionError('newborn taxed on positive cargo')
                home=list(obs['players'][side][1].values());d=lambda p:min(_D[p,b] for b in home)
                mode='stayed' if pre[0]==z else ('homeward' if d(z)<d(pre[0]) else 'other_move')
                counts[side][mode]+=v
                rows.append(dict(source=g['path'],t=t,seat=side,uid=uid,tree=tree,tree_cell=cell,
                                 action=pre[2],land=z,paid=v,mode=mode,history={str(k):past[t-k][side].get(uid) if t>=k else None for k in (0,3,5,10)}))
        assert list(sim.cash)==raw['official_score_by_seat']
        summary.append(dict(source=g['path'],seed=g['seed'],sapling_seat=s,paid_by_seat=[dict(c) for c in counts]))
    sr.retain(HERE/'tax_ledger.json.gz',rows)
    sr.retain(HERE/'tax_totals.json',summary)
    for tag in ('stayed','homeward','other_move'):
        print(tag,[round(sum(r['paid_by_seat'][r['sapling_seat']^j].get(tag,0) for r in summary)/len(summary),2) for j in (0,1)])
if __name__=='__main__':run()

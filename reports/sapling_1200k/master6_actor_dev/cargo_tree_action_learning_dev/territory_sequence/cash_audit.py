"""Retain disaggregated, physical-seat cash accounts and tree census at both horizons."""
from collections import defaultdict
import numpy as np
from data import HERE,D,FastCarbon,CashEvents,sha,read,write,load_game


def run():
    rows=read(HERE/'sequence.json.gz');sources=read(HERE/'sources.json')
    by_game=defaultdict(list)
    for i,r in enumerate(rows):
        by_game[r['game']].append((i,r))
    accounts=np.zeros((len(rows),3,2,6),dtype=np.float64)
    count=np.zeros(len(rows),dtype=np.int16)
    for c in sources:
        cfg,orders,_,initial=load_game(c)
        sim=FastCarbon(cfg).reset(seed=c['seed'],full_carbon=initial)
        sim.cash=CashEvents(sim.cash)
        prefix=np.zeros((218,2,225,3),dtype=np.float64)
        ownership=np.zeros((218,2,225),dtype=np.int16); workers=[]
        for t,joint in enumerate(orders[:217]):
            workers.append(len(sim.player_workers[c['seat']]))
            sim.cash.events.clear();sim.step(*joint)
            prefix[t+1]=prefix[t]
            for kind,recipient,amount,*rest in sim.cash.events:
                if kind=='tree_absorption':
                    prefix[t+1,recipient,rest[1],0]+=amount-rest[2]
                elif kind in ('plant','seize'):
                    prefix[t+1,recipient,rest[1],1]+=amount
                elif kind=='own_delivery':
                    prefix[t+1,recipient,rest[1],2]+=amount
            for tree in sim.trees.values():
                ownership[t+1,tree.player,tree.pos]=1
        for i,r in by_game[c['sha256']]:
            t=r['t'];region=D[r['dst']]<=2;count[i]=workers[t]
            for j,(start,end) in enumerate(((t-8,t),(t,t+4),(t,t+24))):
                vals=prefix[end]-prefix[start]
                dest=accounts[i,j]
                dest[:,:2]=vals[:,region,:2].sum(axis=1)
                dest[:,2]=vals[:,:,2].sum(axis=1)
                dest[:,3]=ownership[start][:,region].sum(axis=1)
                dest[:,4]=ownership[end][:,region].sum(axis=1)
                dest[:,5]=(ownership[end][:,region]*ownership[start][:,region]).sum(axis=1)
    allpred=[]
    for i,r in enumerate(rows):
        s=r['seat'];v=accounts[i,:,s,:]-accounts[i,:,1-s,:]
        prior=v[0,:2].sum()/8
        allpred.append([(v[k,:2].sum()+.25*v[k,2]/count[i])/h-prior for k,h in ((1,4),(2,24))])
    with np.load(HERE/'sequence.npz') as f:
        error=float(np.max(np.abs(allpred-f['y'])))
    assert error<1e-6
    out=HERE/'seat_cash_windows.npz'
    np.savez_compressed(out,accounts=accounts,workers=count)
    write(HERE/'cash_audit.json',dict(windows=['previous_8','next_4','next_24'],axis=['same row as sequence.json.gz','window','physical seat','field'],
                fields=['ground_rent','signed_plant_or_seize_debit','whole_team_own_delivery','own_trees_start','own_trees_end','same_cells_owned_at_both_endpoints'],
                caveat='Endpoint ownership on a cell may include an intervening loss and recapture. Recorded-action destination only.',
                rows=len(rows),max_abs_target_reconstruction_error=error,accounts_sha256=sha(out),row_index_sha256=sha(HERE/'sequence.json.gz')))

if __name__=='__main__':run()

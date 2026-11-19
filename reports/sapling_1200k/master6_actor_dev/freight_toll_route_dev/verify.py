"""Official outcomes, original wall-clock searches, payer receipts and who actually took the tax."""
from response import sr,HERE
import sys
sys.path.insert(0,str(HERE.parent/'two_turn_intercept_dev'))
from ledger import check
from attribution import TollAccount

def run():
    result=[]
    for s in (0,1):
        out=HERE/f'seat{s}';q=sr.load(out/'live_protocol.json');raw=sr.load(sr.ROOT/q['record']['source'])
        focal=q['root']['worker']
        for arm in ('parent','approach'):
            rec=check(out,s,arm);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
            for pair in raw['actions'][:q['root_t']]:sim.step(*pair)
            sim.cash=TollAccount(sim.cash)
            tape=sr.load(out/sr.load(out/f'live_{arm}.json')['action_tape'])
            payments=[]
            for i,cmd in enumerate(tape):
                sim.cash.tolls.clear();sim.step(*cmd)
                for tree,cell,payer,uid,pos,paid in sim.cash.tolls:
                    if uid==focal:payments.append(dict(after=i+1,tree=tree,cell=cell,paid=paid))
            result.append(dict(**rec,focal_tax=payments))
            print(s,arm,rec['score'],round(rec['margin'],2),rec['deposits'][:2],round(sum(r['paid'] for r in payments if r['after']<=5),2))
    sr.retain(HERE/'verified.json',result)
if __name__=='__main__':run()

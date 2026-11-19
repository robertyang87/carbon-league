"""Verify official rewards and trace the two workers, including deposits and local carbon."""
from scan import sr,HERE
import sys
sys.path.insert(0,str(HERE.parent/'two_turn_intercept_dev'))
import ledger
from turning_point_ledger import CashEvents
def run():
    results=[];traces=[]
    for seat in (0,1):
        out=HERE/f'seat{seat}';q=sr.load(out/'live_protocol.json');r=q['root'];raw=sr.load(sr.ROOT/q['record']['source'])
        for arm in ('parent','approach'):
            results.append(ledger.check(out,seat,arm));tape=sr.load(out/f'live_{arm}_actions.json.gz')
            sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed'])
            for cmd in raw['actions'][:r['t']]:sim.step(*cmd)
            sim.cash=CashEvents(sim.cash);history=[]
            for i,cmd in enumerate(tape[:30]):
                obs=sim.obs(seat);a,b=obs['players'][seat],obs['players'][1-seat]
                sim.cash.events.clear();sim.step(*cmd)
                history.append(dict(i=i,carrier=a[2].get(r['worker']),target=b[2].get(r['target']),commands=cmd,
                    visible=obs['carbon'][r['site']],receipts=[e for e in sim.cash.events if e[0]=='own_delivery' and e[3] in (r['worker'],r['target'])]))
            traces.append(dict(seed=q['seed'],seat=seat,arm=arm,history=history))
    sr.retain(HERE/'ledger.json',results);sr.retain(HERE/'worker_history.json.gz',traces)
    for x in results:
        print(x['seat'],x['arm'],x['margin'],x['deposits'][:4],x['windows']['10']['flows'][x['seat']].get('own_delivery',0))
if __name__=='__main__':run()

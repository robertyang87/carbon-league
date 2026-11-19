"""Count actual worker deposits at each stage on both seats of every original-opponent tape."""
from measure import sr,HERE
from turning_point_ledger import CashEvents

def run():
    rows=[]
    for g in sr.load(HERE.parent/'tactical_exchange_dev/paired.json.gz'):
        raw=sr.load(sr.ROOT/g['path']);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=g['seed'])
        sim.cash=CashEvents(sim.cash);deposits=[[[],[]] for _ in range(3)]
        for t,pair in enumerate(raw['actions']):
            sim.cash.events.clear();sim.step(*pair)
            for kind,seat,amount,*detail in sim.cash.events:
                if kind=='own_delivery' and amount>0:
                    deposits[t//100][seat].append(dict(t=t,uid=detail[0],amount=amount))
        assert list(sim.cash)==raw['official_score_by_seat']
        rows.append(dict(source=g['path'],seat=g['seat'],seed=g['seed'],deposits=deposits))
    summary=[]
    for phase in range(3):
        parts=[]
        for r in rows:
            a,b=(r['deposits'][phase][r['seat']^i] for i in range(2))
            c,d=len(a),len(b)
            x,y=(sum(e['amount'] for e in batch)/max(1,len(batch)) for batch in (a,b))
            parts.append([(c-d)*(x+y)/2,(x-y)*(c+d)/2,c*x-d*y,c,d])
        avg=lambda i:sum(p[i] for p in parts)/len(parts)
        summary.append(dict(start=phase*100,frequency=avg(0),amount=avg(1),net=avg(2),sapling_deposits=avg(3),master_deposits=avg(4)))
    sr.retain(HERE/'native_bank_entries.json.gz',rows);sr.retain(HERE/'native_bank_gap.json',summary)
    print(summary)
if __name__=='__main__':run()

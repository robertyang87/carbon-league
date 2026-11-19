"""Evaluate relative receipts for two threatened collectors, once per pair and game."""
from value import HERE, HORIZON, sr, features, predict, _D, CashEvents
import numpy as np


def check(record, weights):
    raw = sr.load(sr.ROOT / record["source"])
    seat = record["seat"]
    sim = sr.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    sim.cash = CashEvents(sim.cash)
    pairs = {}
    payments = {}
    for t, joint in enumerate(raw["actions"]):
        obs = sim.obs(seat)
        own, enemy = obs["players"][seat][2], obs["players"][1-seat][2]
        for a, wa in own.items():
            if wa[2] != "COLLECTOR": continue
            for b, wb in enemy.items():
                if (wb[2] != "COLLECTOR" or wa[1] != wb[1] or _D[wa[0],wb[0]] > 2
                        or (a,b) in pairs or 299-t < HORIZON):
                    continue
                left, right = [float(predict(features(obs, s, uid),weights))
                               for s,uid in ((seat,a),(1-seat,b))]
                pairs[a,b] = dict(t=t, own=a, enemy=b, load=wa[1], predicted=[left,right])
        sim.cash.events.clear()
        sim.step(*joint)
        for kind, _, val, *detail in sim.cash.events:
            if kind=="own_delivery":payments.setdefault(detail[0],[]).append((t+1,val))
        for pair,r in pairs.items():
            if r['t']==t:r['survived']=[u in sim.workers for u in pair]
    assert list(sim.cash)==raw["official_score_by_seat"]
    for pair,r in pairs.items():
        r["actual"] = [sum(v for at,v in payments.get(uid,[]) if r['t']<at<=r['t']+HORIZON) for uid in pair]
    return list(pairs.values())


def main():
    ev=sr.load(HERE/"evaluation.json"); weights=sr.load(HERE/"weights.json")
    raw=[]; summaries=[]
    for g in ev['sources']:
        if g['seed'] not in ev['holdout']:continue
        batch=check(g,weights);raw.extend(dict(**r,seed=g['seed'],seat=g['seat']) for r in batch)
        for tag,group in [('all',batch),('both survived',[r for r in batch if all(r['survived'])])]:
            if not group:continue
            actual=np.array([r['actual'][0]-r['actual'][1] for r in group])
            predicted=np.array([r['predicted'][0]-r['predicted'][1] for r in group])
            nz=actual!=0
            summaries.append(dict(seed=g['seed'],seat=g['seat'],group=tag,n=len(group),unequal=int(nz.sum()),
                     signed_hits=int(np.sum((np.sign(actual)==np.sign(predicted)) & nz)),
                     enemy_higher=int(np.sum(actual<0)),mae=float(np.abs(actual-predicted).mean()),
                     zero_gap_mae=float(np.abs(actual).mean())))
    sr.retain(HERE/"pair_rows.json.gz",raw)
    sr.retain(HERE/"pair_metrics.json",summaries)
    print(summaries)
if __name__=='__main__':main()

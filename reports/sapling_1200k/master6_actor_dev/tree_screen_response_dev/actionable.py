"""Evaluate both forecasts on the held-out opportunities where this pack could change a real order."""
from screen_trial import HERE,sr
def evaluate():
    opportunities=sr.load(HERE.parent/'tactical_exchange_dev/approach.json.gz')
    out=[]
    for planned in (False,True):
        tag='planned_' if planned else ''
        obs=sr.load(HERE/(tag+'threat_observations.json.gz'))
        idx={(r['source'],r['t'],r['tree'],r['foe']):r for r in obs}
        rows=[]
        for g in opportunities:
            r=idx[g['source'],g['t'],g['choice']['tree'],g['choice']['enemy']]
            if 'forecast' in r:
                rows.append({k:r[k] for k in ('source','seed','seat','t','seized','forecast')})
        sr.retain(HERE/(tag+'actionable_forecasts.json'),rows)
        out.append(dict(model=tag+'threat_model',n=len(rows),observed=sum(r['seized'] for r in rows),
                        expected=sum(r['forecast'] for r in rows),range=[min(r['forecast'] for r in rows),max(r['forecast'] for r in rows)]))
    sr.retain(HERE/'actionable_summary.json',out)
    print(out)
if __name__=='__main__':evaluate()

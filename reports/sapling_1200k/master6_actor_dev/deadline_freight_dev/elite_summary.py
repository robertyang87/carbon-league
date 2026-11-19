"""Stratify terminal delivery opportunities by registered player, result, urgency and action."""
import gzip,json
from collections import defaultdict,Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
def summarize():
    report=json.loads(gzip.decompress((HERE/'elite_deadlines.json.gz').read_bytes()))
    groups=defaultdict(Counter); health=Counter();missing=[]
    for g in report['games']:
        if 'error' in g:
            missing.append(g);continue
        health['verified_games']+=1
        if g['statuses']!=['DONE','DONE']:
            health['nonclean_games']+=1;continue
        for r in g['rows']:
            s=r['seat'];diff=g['scores'][s]-g['scores'][1-s]
            outcome='win' if diff>0 else 'loss' if diff<0 else 'draw'
            for key in ((g['names'][s],outcome,r['slack'],r['downhill']),('all','all',r['slack'],r['downhill'])):
                group=groups[key];group['workers']+=1
                group['reached_base']+=r['arrival'] is not None
                group['died_before_return']+=r['death'] is not None
                group['still_away_at_terminal']+=r['arrival'] is None and r['death'] is None
    stats=dict(health=health,missing=missing,
               groups=[dict(player=k[0],result=k[1],slack=k[2],downhill=k[3],**v) for k,v in sorted(groups.items())])
    (HERE/'elite_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':summarize()

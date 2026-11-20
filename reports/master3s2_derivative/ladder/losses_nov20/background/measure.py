"""Reconcile every settled v1 placement at the frozen cutoff; both sides in candidate order."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import math
HERE=Path(__file__).resolve().parent
LADDER=HERE.parent.parent
ROOT=LADDER.parents[2]
sys.path.insert(0,str(ROOT/'reports/sapling_1200k/all_top5'))
from economy_extract import KEYS,reconstruct
sys.path.insert(0,str(LADDER/'tree_payback'))
from labels import tree_rows

def row(card):
    f=LADDER/'replays'/(card['match_id']+'-1.json.gz')
    raw=f.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==card['replay_sha256']
    g=json.loads(gzip.decompress(raw))
    frames=g['steps'];seat=card['candidate_seat']
    physical=[float(x) for x in g['rewards']]
    assert len(frames)-1==card['steps'] and [physical[seat],physical[1-seat]]==card['scores']
    assert abs(physical[seat]-physical[1-seat]-card['official_margin'])<1e-6
    timeline,_,_,_=reconstruct(frames,g['configuration'])
    tree_events=tree_rows(g)
    teams=[]
    for s in (seat,1-seat):
        vals={k:sum(v[s][i] for v in timeline) for i,k in enumerate(KEYS)}
        assert abs(sum(vals.values())-(frames[-1][0]['observation']['players'][s][0]-frames[0][0]['observation']['players'][s][0]))<.015
        own=[t for t in tree_events if t['seat']==s]
        assert abs(sum(t['cost'] for t in own) + vals['plant'])<1e-5
        renew=dict(planted=len(own),first_repaid=sum(t['repaid'] for t in own),first_taken=sum(t['exit']=='seized' for t in own),
            taken_before_payback=sum(t['seized_unrepaid'] for t in own),initial_net=sum(t['net'] for t in own),original_gross=sum(t['gross'] for t in own),
            seized_after_payback=sum(t['exit']=='seized' and t['repaid'] for t in own))
        net=vals['tree']+vals['plant']+vals['seize']
        teams.append(dict(cash={k:round(v,5) for k,v in vals.items()},net_tree=round(net,5),first=renew))
    diffs={k:round(teams[0]['cash'][k]-teams[1]['cash'][k],5) for k in KEYS}
    diffs['net_tree']=round(teams[0]['net_tree']-teams[1]['net_tree'],5)
    assert abs(sum(diffs[k] for k in KEYS)-card['official_margin'])<.025
    return dict(opponent=card['opponent'],match=card['match_id'],seat=seat,official=card['official_margin'],outcome=card['outcome'],
                difference=diffs,candidate_and_opponent=teams)

def summary(group):
    sides=[v['candidate_and_opponent'][0] for v in group]
    fields={k:[v['difference'][k] for v in group] for k in ('bank','net_tree','tree','plant','seize','recruit')}
    fields['official']=[v['official'] for v in group]
    def quant(x,frac):
        y=sorted(x);i=(len(y)-1)*frac;a=int(i);return round(y[a]+(y[min(a+1,len(y)-1)]-y[a])*(i-a),3)
    stats={k:dict(median=quant(v,.5),q1=quant(v,.25),q3=quant(v,.75),mean=round(sum(v)/len(v),3),
                  positive=sum(n>0 for n in v),negative=sum(n<0 for n in v)) for k,v in fields.items()}
    return dict(games=len(group),stats=stats,
        signs_bank_then_tree=dict(Counter(('+' if r['difference']['bank']>0 else '-')+('+' if r['difference']['net_tree']>0 else '-') for r in group)),
        candidate_pooled_trees={k:sum(s['first'][k] for s in sides) for k in ('planted','first_repaid','first_taken','taken_before_payback','seized_after_payback')},
        opposite_pooled_trees={k:sum(v['candidate_and_opponent'][1]['first'][k] for v in group) for k in ('planted','first_repaid','first_taken','taken_before_payback','seized_after_payback')},
        per_game_candidates_paid=dict(median_plants=median([s['first']['planted'] for s in sides]),
            median_plant_spend=round(median([-s['cash']['plant'] for s in sides]),2),median_seize_spend=round(median([-s['cash']['seize'] for s in sides]),2),
            median_payback_fraction=round(median([s['first']['first_repaid']/s['first']['planted'] for s in sides if s['first']['planted']]),4)),
        seats=dict(Counter(v['seat'] for v in group)))

if __name__=='__main__':
    reg=json.loads((HERE/'manifest.json').read_bytes())
    games=[row(c) for c in reg['games']]
    assert len(set(v['match'] for v in games))==31 and len(set(c['map_sha256'] for c in reg['games']))==31
    focus=set(reg['focus_ids'])
    groups={'wins':[r for r in games if r['outcome']=='win'],'focused_losses':[r for r in games if r['match'] in focus],
        'other_losses':[r for r in games if r['outcome']=='loss' and r['match'] not in focus]}
    assert [len(x) for x in groups.values()]==[26,3,2]
    out=dict(manifest_sha256=hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest(),
        teams_order=['ds-pro-v1','opponent'],net_tree='tree absorption less planting and seizure outlays; direct collision credited separately',
        summaries={key:summary(value) for key,value in groups.items()},games=games)
    raw=(json.dumps(out,allow_nan=False,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
    (HERE/'results.json').write_bytes(raw)
    print(hashlib.sha256(raw).hexdigest())

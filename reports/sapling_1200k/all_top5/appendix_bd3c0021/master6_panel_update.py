"""Extend the sealed bf049043 cash and economic-stop comparison with one newer appendix."""
from collections import Counter
from pathlib import Path
import gzip
import json
import sys

OUT=Path(__file__).resolve().parent
HERE=OUT.parent
PRIOR=HERE/'appendix_bf049043'
sys.path.insert(0,str(PRIOR))
import master6_panel_update as panel
from master6_recovery import ARMS, analyze
from context import Corpus

def rebuild(output=OUT):
    places=[HERE/'appendix_555c0c56',PRIOR,OUT]
    cards,source,offsets,seals,raw,excluded,when,_=panel.take(places)
    before=panel.load(PRIOR/'master6_panel_update.json')
    original=[r for r in raw if source[r['id']]!=OUT.name]
    oldecon,oldstrata=panel.compact_panel(original)
    if oldecon!=before['economic_panel']:raise ValueError('prior economic means or strata changed')
    summary,strata=panel.compact_panel(raw)
    old_strict=panel.load(HERE/'master6_recovery.json')['maps']
    old_strict.extend(panel.load(PRIOR/'master6_recovery_new_rows.json.gz'))
    records={r['id']:r for r in old_strict}
    oldmembers={gid for arms in oldstrata.values() for arm in arms for gid in arm}
    if oldmembers!=set(records) or len(records)!=sum(oldecon['matched_maps']):raise ValueError('missing prior records')
    if panel.rates([records[k] for k in sorted(records)])!=before['strict_recovery']:raise ValueError('prior recovery cache changed')
    for gid,record in records.items():
        if record['replay_sha256']!=cards[gid]['sha256']:raise ValueError('prior replay was replaced')
    memberships={gid:(name,seat,arm) for (name,seat),arms in strata.items() for arm,ids in zip(ARMS,arms) for gid in ids}
    if len(memberships)!=sum(summary['matched_maps']) or not oldmembers<=memberships.keys():raise ValueError('non-monotonic or duplicated panel')
    corpora={};new_rows=[]
    for gid in sorted(memberships.keys()-oldmembers):
        name,seat,arm=memberships[gid];batch=source[gid]
        if batch not in corpora:
            corpora[batch]=Corpus() if batch=='base' else Corpus(manifest=HERE/batch/'manifest.json')
        c=corpora[batch];meta=c.provenance(gid)
        if meta['seat_names'][seat]!=name or (meta['seat_names'][1-seat]==panel.ctl.MASTER)!=(arm==ARMS[0]):raise ValueError('stratum role changed')
        r=analyze(c,gid,seat,name,arm)
        if r['replay_sha256']!=cards[gid]['sha256']:raise ValueError('recovery record belongs to wrong replay')
        records[gid]=r;new_rows.append(r)
    ordered=[records[k] for k in sorted(records)];recovery=panel.rates(ordered);cluster=panel.rate_interval(ordered)
    leaders=panel.load(HERE.parent/'latest5_inventory_sources.json.gz')['leaderboard']
    regs={q['id']:q for q in leaders['strategies']+leaders['extra']};names=Counter(q['name'] for q in regs.values())
    ranks={q['name']:q['board_rank'] for q in regs.values() if names[q['name']]==1 and q.get('board')=='heavy' and isinstance(q.get('board_rank'),int)}
    newcards=[]
    for card in panel.load(OUT/'manifest.json')['games']:
        gid=card['id'];s=card['seat_strategy_ids'].index(None) if card['cohort']=='external' else None
        name=card['seat_names'][s] if s is not None else None;rank=ranks.get(name)
        status=('matched' if gid in memberships else 'top5_mutual' if card['cohort']!='external' else
                'unclean' if not card['clean_terminal'] else 'external_identity_or_rank_unresolved' if rank is None else
                'rank_outside_21_50' if not 21<=rank<=50 else 'no_matching_arm_at_this_seat')
        newcards.append(dict(id=gid,seed=card['seed'],players=card['seat_names'],seat=s,rank=rank,disposition=status,
                             arm=memberships[gid][2] if gid in memberships else None,
                             strict=bool(records[gid]['runs']['strict']) if gid in memberships else None))
    changes=[]
    for key,arms in sorted(strata.items()):
        previous=oldstrata.get(key,[[],[]]);newids=[sorted(set(v)-set(prior)) for v,prior in zip(arms,previous)]
        if any(newids):
            changes.append(dict(opponent=key[0],seat=key[1],before=list(map(len,previous)),after=list(map(len,arms)),
                                added=[[{'id':g,'source':source[g]} for g in arm] for arm in newids]))
    files=[PRIOR/'master6_panel_update.json',PRIOR/'master6_recovery_new_rows.json.gz',PRIOR/'master6_panel_update.py',
           HERE/'master6_recovery.json', HERE/'master6_opponent_controls.py',HERE/'master6_recovery.py',HERE/'context.py',Path(__file__)]
    hashes={str(p.relative_to(HERE)):panel.sha(p) for p in files}
    data={'sealed_after':offsets[-1], 'rank_snapshot_utc':when, 'previous':{'coverage':before['coverage'][-1],'matched_maps':oldecon['matched_maps'],
          'strict_games':[before['strict_recovery'][a]['flags']['strict']['1']['games'] for a in ARMS]},
          'metric_order':panel.ctl.METRICS,'stage_intervals':panel.ctl.PHASES,'arms':ARMS,'cumulative_exclusions':dict(excluded),
          'economic_panel':summary,'strict_recovery':recovery,'strict_matched_rate_interval':cluster,'appended_games':newcards,'changed_strata':changes,
          'newly_promoted_older_games':[r['id'] for r in new_rows if source[r['id']]!=OUT.name],'cached_maps':len(oldmembers),'newly_scanned_maps':len(new_rows),
          'appendix_inputs':seals,'prior_cache_and_code':hashes}
    panel.dump(output/'master6_panel_update.json',data)
    (output/'master6_recovery_new_rows.json.gz').write_bytes(gzip.compress(json.dumps(new_rows,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode(),mtime=0))
    return data
if __name__=='__main__':
    s=rebuild();print(s['economic_panel']['matched_maps'],[s['strict_recovery'][a]['flags']['strict']['1']['games'] for a in ARMS])

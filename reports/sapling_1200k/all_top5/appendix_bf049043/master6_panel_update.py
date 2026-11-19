"""Update the sealed 1,347-game Master6 comparisons with consecutive, disjoint appendix batches."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import master6_opponent_controls as ctl
import master6_recovery as rec
from context import Corpus

OLD=HERE/'appendix_4bfdabf5'; EARLY=HERE/'appendix_499886f1'
def load(p):return json.loads(gzip.decompress(p.read_bytes()) if p.name.endswith('.gz') else p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,o):path.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def take(places):
    baseline=load(HERE/'manifest.json');cards={g['id']:g for g in baseline['games']};source={g:'base' for g in cards}
    seeds={g['seed'] for g in cards.values()};allrevisions=[];seals={}
    cumulative=sorted([EARLY,OLD,*places],key=lambda p:load(p/'manifest.json')['baseline_games'])
    if cumulative[:2]!=[EARLY,OLD] or cumulative[2:]!=places:raise ValueError('unordered appendix set')
    for where in cumulative:
        man=load(where/'manifest.json');p=where/'cases.json.gz';cases=load(p)
        if man['baseline_games']!=len(cards) or man['revision']!=cases['revision'] or {g['id'] for g in man['games']}!={r['id'] for r in cases['rows']}: raise ValueError(f'nonconsecutive or incomplete appendix {where}')
        seals[where.name]={q.name:sha(q) for q in (p,where/'manifest.json')}
        for g in man['games']:
            if g['id'] in cards or g['seed'] in seeds:raise ValueError('reused ID or seed')
            cards[g['id']]=g;source[g['id']]=where.name;seeds.add(g['seed'])
        if man['accumulated_games']!=len(cards):raise ValueError('cumulative count')
        allrevisions.append([where.name,man['revision'],len(man['games']),len(cards)])
    # The historical loader accepts one extra capsule. Present the already verified
    # appendices as one temporary read-only concatenation, then reuse the frozen logic.
    concat=cumulative[1:];extra_games=[];extra_rows=[]
    for where in concat:
        extra_games.extend(load(where/'manifest.json')['games']);extra_rows.extend(load(where/'cases.json.gz')['rows'])
    with tempfile.TemporaryDirectory(prefix='.matched-panel-',dir=HERE) as tmpname:
        tmp=Path(tmpname);dump(tmp/'manifest.json',{'games':extra_games})
        (tmp/'cases.json.gz').write_bytes(gzip.compress(json.dumps({'rows':extra_rows},ensure_ascii=False).encode(),mtime=0))
        raw,exclusions,_,when,maxrank=ctl.load(tmp)
    return cards,source,allrevisions,seals,raw,exclusions,when,maxrank

def compact_panel(raw):
    full=ctl.summarize(raw,21,50,bootstrap=True,examples=True)
    s=full.pop('strata')
    strata={(r['opponent'],r['seat']):r['arm_ids'] for r in s}
    for key in ('counterexample','suppressed_example'):full.pop(key,None)
    full['by_seat']={str(i):ctl.summarize(raw,21,50,physical=i) for i in (0,1)}
    full['fixed_comparator']={k:ctl.summarize(raw,21,50,control=k) for k in sorted({x['head'] for x in raw}-{ctl.MASTER})}
    full['survived_to_200']=ctl.summarize(raw,21,50,complete=True)
    return full,strata
def rates(rows):
    result=rec.summary(rows)
    for arm,z in result.items():
        players=[q for q in rows if q['arm']==arm]
        regained=[(r,e) for r in players for e in r['runs']['gap_after_collector'] if e['collector_returned_at_end']]
        lost=[(r,e) for r in players for e in r['runs']['strict']]
        z['regained_collector_episodes']=len(regained)
        z['regained_collector_games']=len({r['id'] for r,e in regained})
        z['strict_at_terminal_games']=len({r['id'] for r,e in lost if e['strict_at_terminal']})
        z['strict_income_abs_max']=max((max(e['gross_cash_event_absolutes'].values(),default=0.) for _,e in lost),default=0.)
    return result

def rate_interval(rows):
    by=defaultdict(lambda:defaultdict(list))
    for r in rows:
        by[r['opponent'],r['seat']][r['arm']].append(bool(r['runs']['strict']))
    people=defaultdict(list)
    for (name,_),arms in sorted(by.items()):
        if any(a not in arms for a in rec.ARMS):raise ValueError('unmatched rate stratum')
        people[name].append([np.mean(arms[a]) for a in rec.ARMS])
    matrix=np.array([np.mean(people[name],axis=0) for name in sorted(people)])
    differences=matrix[:,0]-matrix[:,1]
    rng=np.random.default_rng(20261119)
    estimates=differences[rng.integers(len(people),size=(10000,len(people)))].mean(axis=1)
    return {'unit':'external registration, equal eligible seat weights; fixed matched games; resample identities',
            'replicates':len(estimates),'rate_by_arm':matrix.mean(axis=0).tolist(),
            'difference':float(differences.mean()),'percentile_95':np.quantile(estimates,[.025,.975]).tolist()}

def update(places, out=None):
    cards,source,offsets,seals,raw,exclude,when,high=take(places);out=out or places[-1]
    registered=load(OLD/'master6_opponent_controls.json')['groups']['21_50']
    oldkeys={(r['opponent'],r['seat']):r['arm_ids'] for r in registered['strata']}
    oldraw=[r for r in raw if source[r['id']] not in {p.name for p in places}]
    recheck,prev=compact_panel(oldraw)
    if oldkeys!=prev or any(recheck[k]!=registered[k] for k in recheck):raise ValueError('prior economic panel failed independent readback')
    historic=load(HERE/'master6_recovery.json');oldmaps={r['id']:r for r in historic['maps']}
    seen=set(oldmaps);econ_by={r['id']:r for r in raw}
    new_by_source={where.name:{'new_cards':[], 'panel_new_ids':[], 'promoted_older_ids':[]} for where in places}
    catalog_dirs=[HERE,EARLY,OLD,*places]; readers={}
    def corpus(gid):
        d=next(k for k in catalog_dirs if k.name==source[gid]) if source[gid]!='base' else HERE
        if d not in readers:readers[d]=Corpus(manifest=d/'manifest.json') if d!=HERE else Corpus()
        return readers[d]
    recovery=dict(oldmaps);snapshots=[]
    board=load(HERE.parent/'latest5_inventory_sources.json.gz')['leaderboard']
    reg={q['id']:q for q in board['strategies']+board['extra']};reg_names=Counter(q['name'] for q in reg.values())
    ranks={q['name']:q['board_rank'] for q in reg.values() if reg_names[q['name']]==1 and q.get('board')=='heavy' and isinstance(q.get('board_rank'),int)}
    if rates(list(recovery.values()))[rec.ARMS[0]]['flags']['strict']['1']['games']!=9:raise ValueError('strict baseline')
    for i,where in enumerate(places):
        included={w.name for w in places[:i+1]};excluded={w.name for w in places}-included
        current=[r for r in raw if source[r['id']] not in excluded]
        summary,strata=compact_panel(current)
        expected={g:(name,s,arm) for (name,s),group in strata.items() for arm,ids in zip(rec.ARMS,group) for g in ids}
        if not seen<=set(expected):raise ValueError('lost a prior matched map')
        count_before=len(seen);new_ids=set(expected)-seen;changes=[]
        for gid in sorted(new_ids):
            name,seat,arm=expected[gid];co=corpus(gid);meta=co.provenance(gid)
            if meta['seat_names'][seat]!=name or (meta['seat_names'][1-seat]==ctl.MASTER)!=(arm==rec.ARMS[0]):raise ValueError('identity mismatch')
            result=rec.analyze(co,gid,seat,name,arm)
            if result['replay_sha256']!=cards[gid]['sha256']:raise ValueError('new strict replay identity')
            recovery[gid]=result;bucket='panel_new_ids' if source[gid]==where.name else 'promoted_older_ids'
            new_by_source[where.name][bucket].append(gid);seen.add(gid)
        if len(seen)!=sum(summary['matched_maps']):raise ValueError('wrong new panel count')
        for key,arms in sorted(strata.items()):
            prior=prev.get(key,[[],[]]);extra=[sorted(set(arm)-set(prior[a])) for a,arm in enumerate(arms)]
            if any(extra):changes.append({'name':key[0],'physical_seat':key[1],'prior_counts':[len(a) for a in prior],'current_counts':[len(a) for a in arms],'new_ids_by_arm':extra})
        current_recovery=rates([recovery[k] for k in sorted(expected)])
        new_ids_only=load(where/'manifest.json')['games']
        for card in new_ids_only:
            gid=card['id'];s=None if card['cohort']!='external' else card['seat_strategy_ids'].index(None)
            n=card['seat_names'][s] if s is not None else None;r=ranks.get(n)
            if gid in expected:
                status='in_matched_panel'
            elif card['cohort']!='external':status='mutual_top5'
            elif not card['clean_terminal']:status='unclean'
            elif r is None:status='identity_or_rank_unresolved'
            elif not 21<=r<=50:status='outside_rank_21_50'
            else:status='no_common_opponent_and_seat'
            new_by_source[where.name]['new_cards'].append({'id':gid,'seed':card['seed'],'seat_names':card['seat_names'],'external_seat':s,'external_rank':r,
                'panel':status,'arm':expected[gid][2] if gid in expected else None,
                'strict_during_game':bool(recovery[gid]['runs']['strict']) if gid in expected else None})
        snapshots.append({'after':where.name,'accumulated_games':load(where/'manifest.json')['accumulated_games'],'matched_maps':summary['matched_maps'],
                          'matched_opponents':summary['matched_opponents'],'matched_seat_strata':summary['matched_opponent_seats'],
                          'strict_games':[current_recovery[a]['flags']['strict']['1']['games'] for a in rec.ARMS],'new_strata_or_maps':changes})
        prev=strata
    final={'rank_snapshot_utc':when,'coverage':offsets,'raw_exclusions':dict(exclude),'baseline_map_count':len(oldmaps),'metric_order':ctl.METRICS,'phase_intervals':ctl.PHASES,
           'arm_order':list(rec.ARMS),'arm_contrast':'All values from matched external player perspective, facing Master6 minus the other frozen top-five players. Identity equal weighted, seats equal within identity.',
           'economic_panel':summary,'strict_matched_rate_interval':rate_interval([recovery[k] for k in sorted(expected)]),'strict_recovery':current_recovery,'snapshots':snapshots,'appended_game_eligibility':new_by_source,
           'baseline_files':{p.name:sha(p) for p in (OLD/'master6_opponent_controls.json',HERE/'master6_recovery.json')},
           'source_hashes':{p.name:sha(p) for p in (HERE/'master6_opponent_controls.py',HERE/'master6_recovery.py',HERE/'context.py',Path(__file__),HERE.parent/'latest5_inventory_sources.json.gz')},'appendix_inputs':seals}
    dump(out/'master6_panel_update.json',final)
    extra_rows=[r for gid,r in sorted(recovery.items()) if gid not in oldmaps]
    (out/'master6_recovery_new_rows.json.gz').write_bytes(gzip.compress(json.dumps(extra_rows,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    print([(p['after'],p['matched_maps'],p['strict_games']) for p in snapshots])
if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('appendix',nargs='+',help='ordered sealed appendix directories, starting after 4bfdabf5')
    args=cli.parse_args();update([HERE/p for p in args.appendix])

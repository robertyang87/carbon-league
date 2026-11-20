"""Read-only ladder census and official replay reconciliation for ds-pro-v4."""
import argparse
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'placement_refresh'))
import refresh
from report import report
ID='16a3cd313829ffe78bd24ac9714b2400'
refresh.ID=ID
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def history(frames,events,t):
    found=[]
    for e in events:
        seat=e['seat']; uid=e['worker'];end=e['arrival']
        route=[]
        for before in sorted(set(max(0,end-d) for d in (10,5,3,1,0))):
            val=frames[before][0]['observation']['players'][seat][2].get(uid)
            cmd=(frames[before+1][seat]['action'] or {}).get(uid,'NONE') if before<299 and val else None
            route.append(dict(step=before,worker_state=val,issued=cmd))
        found.append(e|dict(route=route))
    return found


def turning(game,replay):
    seat=game['seat'];frames=replay['steps'];states=game['steps'];m=[r['margin'] for r in states]
    team=1 if m[-1]>0 else -1
    qty=250;length=20
    def first(sgn):
        return next((i for i in range(1,len(m)-length+1) if all(x*sgn>=qty for x in m[i:i+length])),None)
    points={'lead':first(1),'deficit':first(-1)}
    eligible=[k for k in points.values() if k is not None]
    t=min(eligible) if eligible else None
    timeline,_,_,_=refresh.compare.economy.reconstruct(frames,replay['configuration'])
    def evidence(t):
        cards=[]
        for w in (3,5,10):
            a=max(0,t-w)
            events=[e|{'arrival':k+1} for k in range(a,t) for e in game['receipts_by_transition'][k]]
            totals=[sum(e['amount'] for e in events if e['seat']==j) for j in (seat,1-seat)]
            # Show the two largest bank events from each physical side to capture offsetting moves.
            important=[e for j in (seat,1-seat) for e in sorted([q for q in events if q['seat']==j],key=lambda q:-q['amount'])[:2]]
            incomes,diffs,_=refresh.review.cash_accounts(timeline[a:t],seat)
            new_plants=[v for group in game['planting'] for v in group['individual'] if a<=v['t']<t]
            cards.append(dict(start=a,end=t,margin_before=m[a],margin_after=m[t],bank_by_entrant=totals,ending_state=states[t],
                              incomes=incomes,differences=diffs,new_plant_events=new_plants,
                              deposits_with_action_history=history(frames,important,t),
                              joint_orders_at_start=game['full_joint_commands'][a],joint_orders_into_crossing=game['full_joint_commands'][t-1]))
        return cards
    final_t=next((i for i in range(1,len(m)) if m[i]*team>0 and min(x*team for x in m[i:])>0),None)
    early=next((i for i in range(5,len(m)) if abs(m[i]-m[i-5])>=200),None)
    return dict(schema=2,definition="Earliest absolute 200-point five-step swing; first 250-point lead/deficit lasting twenty frames; earliest permanent final sign. All defined from v4 canonical seat.",
                first_five_step_swing=early,early=evidence(early) if early else [],
                first_durable_lead=points['lead'],first_durable_deficit=points['deficit'],first_crossing=t,
                crossing=evidence(t) if t else [],permanent_final_sign=final_t,ending=evidence(final_t) if final_t else [],
                worst_ten_for_us=game['worst_windows']['10'],score=game['official_scores'])


def historical(opponents):
    refs={}
    ident=refresh.ID
    try:
        for ver in ('v1','v2'):
            identity=refresh.read(HERE.parent/f'ds-pro-{ver}.json')['platform']['id']
            source=HERE.parent/f'nn_patch_cutoff/{ver}_pages.json.gz'
            refresh.ID=identity
            for p in refresh.read(source):
                for m in p['data']['matches']:
                    if m['state']!='done':continue
                    for g in m['result']:
                        row=refresh.score(m,g)
                        if row['opponent_id'] in opponents:
                            refs[(row['match'],row['game'])]=dict(version=ver,**row)
    finally:refresh.ID=ident
    return sorted(refs.values(),key=lambda r:r['created'])


def run(new_snapshot):
    pages=HERE/'pages.json.gz'
    if new_snapshot:
        fresh=HERE/'pages.next.json.gz';refresh.fetch(fresh)
        fresh.replace(pages)
    raw=refresh.read(pages)
    matches={m['id']:m for pg in raw for m in pg['data']['matches']}
    assert len(matches)==sum(len(pg['data']['matches']) for pg in raw)
    games=[];pending=[];seen=set()
    for m in sorted(matches.values(),key=lambda v:v['created']):
        if m['state']!='done' or not m['result']:
            pending.append({k:m.get(k) for k in ('id','state','kind','name_a','name_b','created')});continue
        for g in m['result']:
            c=refresh.score(m,g); side=g['order'][c['seat']];op='b' if side=='a' else 'a'
            c['timing_by_entrant']=[g['action_timing'][k] for k in (side,op)] if g.get('action_timing') else None
            key=(c['match'],c['game']);assert key not in seen;seen.add(key)
            fn=f"{key[0]}-{key[1]}";archive=HERE/'replays'/(fn+'.json.gz');ledger=HERE/(fn+'.ledger.json.gz')
            if not archive.exists():
                archive.parent.mkdir(exist_ok=True)
                env,_,_=refresh.extract(refresh.get(f"{refresh.BASE}/replay/{key[0]}/{key[1]}"))
                refresh.validate(env,{**m,**g})
                archive.write_bytes(gzip.compress(refresh.canonical(env),compresslevel=6,mtime=0))
            env=refresh.read(archive)
            c['validation']=refresh.validate(env,{**m,**g});assert c['validation']['scores_checked'] and c['validation']['steps_checked']
            c['terrain']=sha(bytes(env['steps'][0][0]['observation']['full_carbon']))
            c['archive_sha256']=sha(archive.read_bytes());c['replay']=str(archive.relative_to(HERE))
            if c['steps']==299 and list(env['statuses'])==['DONE','DONE']:
                refresh.compare.HERE=HERE/'analysis'
                if ledger.exists():
                    d=refresh.read(ledger)
                    if d.get('turning',{}).get('schema')!=2:
                        d['turning']=turning(d,env);refresh.store(ledger,d)
                else:
                    d=refresh.review.losing_game(dict(opponent=c['opponent'],match_id=c['match'],seat=c['seat'],steps=c['steps']),archive)
                    d['turning']=turning(d,env)
                    refresh.store(ledger,d)
                c['official_reproduced']=d['official'];c['cash_account']=d['whole']['difference'];c['ledger']=ledger.name
                q=d['turning'];c['first_durable_lead']=q['first_durable_lead'];c['first_durable_deficit']=q['first_durable_deficit']
                c['permanent_final_sign']=q['permanent_final_sign']
            games.append(c)
    result=dict(strategy=ID,cutoff_utc=raw[0]['fetched_utc'],last_page_utc=raw[-1]['fetched_utc'],pages_sha256=sha(pages.read_bytes()),counts=dict(Counter(c['outcome'] for c in games)),
                by_kind={k:dict(Counter(c['outcome'] for c in games if c['kind']==k)) for k in sorted({c['kind'] for c in games})},pending=pending,
                unique_terrains=len({c['terrain'] for c in games}),games=games)
    result['registered_opponent_comparisons']=historical({c['opponent_id'] for c in games})
    refresh.store(HERE/'summary.json',result)
    report(result)
    print(result['cutoff_utc'],result['counts'],len(pending),[(c['opponent'],c['seat'],c['margin']) for c in games],flush=True)

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--refresh',action='store_true');args=cli.parse_args();run(args.refresh)

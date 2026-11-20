"""Incremental official reconciliation; previously archived four results remain immutable."""
import gzip
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from observe import refresh,turning,sha

def main():
    prev=refresh.read(HERE.parent/'summary.json')
    reference={(c['match'],c['game']):c for c in prev['games']}
    pages=refresh.read(HERE/'pages.json.gz');seen=set();games=[];pending=[]
    for m in [m for p in pages for m in p['data']['matches']]:
        if m['state']!='done':
            pending.append({k:m[k] for k in ('id','name_a','name_b','state')});continue
        for g in m['result']:
            c=refresh.score(m,g);k=(c['match'],c['game'])
            assert k not in seen;seen.add(k)
            if k in reference:
                assert all(c[f]==reference[k][f] for f in ('score','margin','opponent_id','seat','outcome'))
                continue
            key=f'{k[0]}-{k[1]}';archive=HERE/'replays'/f'{key}.json.gz'
            if not archive.exists():
                game,_,_=refresh.extract(refresh.get(f'{refresh.BASE}/replay/{k[0]}/{k[1]}'))
                refresh.validate(game,{**m,**g})
                archive.write_bytes(gzip.compress(refresh.canonical(game),compresslevel=6,mtime=0))
            game=refresh.read(archive);val=refresh.validate(game,{**m,**g})
            c.update(terrain=sha(bytes(game['steps'][0][0]['observation']['full_carbon'])),replay_sha256=sha(archive.read_bytes()),
                     replay=str(archive.relative_to(HERE)),timing=g.get('action_timing'),validated=val)
            if c['steps']==299 and game['statuses']==['DONE','DONE']:
                refresh.compare.HERE=HERE/'analysis'
                d=refresh.review.losing_game(dict(opponent=c['opponent'],match_id=c['match'],seat=c['seat'],steps=c['steps']),archive)
                d['turning']=turning(d,game);fn=HERE/f'{key}.ledger.json.gz';refresh.store(fn,d)
                c.update(ledger=fn.name,ledger_sha256=sha(fn.read_bytes()),official=d['official'],cash=d['whole']['difference'])
            games.append(c)
    assert set(reference)<=seen
    out=dict(cutoff=pages[-1]['fetched_utc'],prior_games_verified=len(reference),added=games,pending=pending,
             games_total=len(seen),pages_sha256=sha((HERE/'pages.json.gz').read_bytes()))
    refresh.store(HERE/'summary.json',out)
    print(out['cutoff'],out['games_total'],[(c['opponent'],c['margin'],c.get('cash')) for c in games])
if __name__=='__main__':main()

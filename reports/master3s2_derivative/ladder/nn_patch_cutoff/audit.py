"""One public, read-only scorecard census followed by validation of newly settled losses."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
LADDER=HERE.parent
sys.path.insert(0,str(LADDER/'post_v3_reconciliation'))
import matches
from matches import refresh,cutoff

def audit(download):
    scores=refresh.read(HERE/'matches.json')
    census={}
    for version,summary in scores['versions'].items():
        refresh.ID=summary['id']
        matches_by_id={m['id']:m for page in refresh.read(HERE/f'{version}_pages.json.gz') for m in page['data']['matches']}
        prior=refresh.read(LADDER/'pre_v3_cutoff/summary.json')['strategies'][version]
        previous_pending={a['id'] for a in prior['pending']}
        ledgers=[]
        archives=[]
        for c in summary['added']:
            match=matches_by_id[c['match']]
            game=next(g for g in match['result'] if g['game']==c['game'])
            assert refresh.score(match,game)==c
            name=f"{c['match']}-{c['game']}.json.gz"
            saved=next(iter(sorted(path for path in LADDER.glob('**/'+name) if 'replays' in path.parts)),None)
            if saved is None:
                if not download:raise FileNotFoundError(name)
                saved=HERE/'replays'/name
                saved.parent.mkdir(exist_ok=True)
                replay,_,_=cutoff.extract(refresh.get(f"{refresh.BASE}/replay/{c['match']}/{c['game']}"))
                refresh.validate(replay,{**match,**game})
                saved.write_bytes(gzip.compress(cutoff.canonical(replay),compresslevel=6,mtime=0))
            archive=refresh.read(saved)
            validate=refresh.validate(archive,{**match,**game})
            record=dict(match=c['match'],game=c['game'],opponent=c['opponent'],score=c['score'],margin=c['margin'],
                           seat=c['seat'],source=f"{refresh.BASE}/replay/{c['match']}/{c['game']}",validation=validate,
                           terrain=hashlib.sha256(bytes(archive['steps'][0][0]['observation']['full_carbon'])).hexdigest(),
                           replay=str(saved.relative_to(LADDER)),replay_sha256=cutoff.digest(saved.read_bytes()))
            archives.append(record)
            paired_win=(version=='v2' and c['match'] in previous_pending)
            if c['outcome']!='L' and not paired_win:continue
            assert c['steps']==299 and all(v=='DONE' for v in c['statuses'].values()), 'Partial game: retain as exceptional outcome'
            refresh.compare.HERE=saved.parent.parent/'analysis'
            detail=refresh.review.losing_game(dict(opponent=c['opponent'],match_id=c['match'],seat=c['seat'],steps=c['steps']),saved)
            fn=HERE/f"{version}_{c['match']}-{c['game']}.ledger.json.gz"
            refresh.store(fn,detail)
            record['ledger']=fn.name
            record['ledger_sha256']=cutoff.digest(fn.read_bytes())
            if c['outcome']=='L':ledgers.append(record)
            print(version,c['opponent'],c['match'],flush=True)
        census[version]=dict(prior_games=summary['verified_prior'],losses=ledgers,all_new_replays=archives,
                previously_pending_now_settled=[c for c in summary['added'] if c['match'] in previous_pending])
    refresh.store(HERE/'audit.json',census)

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--fetch',action='store_true')
    cli.add_argument('--download',action='store_true')
    opt=cli.parse_args()
    matches.HERE=HERE
    matches.run(opt.fetch)
    audit(opt.download)

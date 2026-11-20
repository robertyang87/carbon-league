"""Read official v4 match cards and retrieve the completed d5 opponent replay."""
import gzip
import sys
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'placement_refresh'))
import refresh
refresh.ID='16a3cd313829ffe78bd24ac9714b2400'
if __name__=='__main__':
    refresh.fetch(HERE/'v4-pages.json.gz')
    matches=[m for p in refresh.read(HERE/'v4-pages.json.gz') for m in p['data']['matches']]
    for m in matches:
        if not any('v5-d5' in (m['name_'+s] or '') and '小肥鱼' in (m['name_'+s] or '') for s in ('a','b')): continue
        print(m['state'],m['id'],[m['name_a'],m['name_b']],flush=True)
        if m['state']!='done' or not m['result']:continue
        for g in m['result']:
            c=refresh.score(m,g)
            print(c,flush=True)
            h=HERE/'replays'/f"{c['match']}-{c['game']}.json.gz"
            env,_,_=refresh.extract(refresh.get(f"{refresh.BASE}/replay/{c['match']}/{c['game']}"))
            c['official_validation']=refresh.validate(env,{**m,**g})
            assert c['official_validation']['scores_checked'] and c['official_validation']['steps_checked']
            h.write_bytes(gzip.compress(refresh.canonical(env),compresslevel=6,mtime=0))
            c['sha256']=hashlib.sha256(h.read_bytes()).hexdigest()
            c['opponent_registration']=m['b' if m['a']==refresh.ID else 'a']
            c['replay_file']=str(h.relative_to(HERE))
            refresh.store(HERE/'match.json',c)

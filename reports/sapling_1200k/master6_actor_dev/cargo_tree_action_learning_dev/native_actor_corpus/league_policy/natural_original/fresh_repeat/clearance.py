"""Seal two replicated original-opponent maps against the frozen repository and latest actual ladder openings."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import io
import zipfile
import sys

HERE=Path(__file__).resolve().parent
LEAGUE=HERE.parents[1]
SAP=next(p for p in HERE.parents if (p/'map_identity.py').is_file())
TOP=SAP/'all_top5'
spec=importlib.util.spec_from_file_location('league_clearance_census',LEAGUE/'clearance.py')
registry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)
SNAPSHOT='b48913c55f4ed816a1b5305669ca3e4dd61f4d81'
FINAL='1321704d68b335b9a6baebc313071d0e55f3c571'
SCORE_BASE='628e28d06b0610f6ceebdac3cc632057766bddd2'
SEEDS=[621758450,621758451]
sys.path.insert(0,str(TOP))
import fetch
from appendix_97de1ebc.collect import all_sources, read_score

def full_census():
    registry.SNAPSHOT=SNAPSHOT
    registry.GROUPS={'new_original_development':SEEDS}
    registry.SEEDS=SEEDS
    registry.CANDIDATE=re.compile(rb'(?<![.\d])(?:621758450|621758451)(?![.\d])')
    result=registry.scan()
    result.update(snapshot=SNAPSHOT,seeds=SEEDS,purpose='Two first-use development maps for sampled parent 93d70be4 against sampled child 851c901a, each versus original Master3s2. Independently registered formal and earlier 444/445 maps remain reserved.')
    return result

def significant_cards(rev):
    paths=all_sources(rev); cards={}; total=0
    for p in sorted(paths):
        for k,r in read_score(p,rev,paths).items():
            total+=1
            seats=tuple(r['name_'+s] for s in r['order']);scores=tuple(r['scores'].get(s) for s in r['order'])
            if k in cards and cards[k]['identity']!=(seats,scores):
                raise ValueError('Conflicting score-card mirrors '+k)
            cards[k]={'identity':(seats,scores),'card':r}
    return cards,dict(indices=len(paths),references=total,distinct=len(cards))

def score_census():
    old,old_audit=significant_cards(SCORE_BASE);new,new_audit=significant_cards(SNAPSHOT)
    if old.keys()-new.keys() or any(new[k]['identity']!=v['identity'] for k,v in old.items()):
        raise ValueError('Previous published score card changed')
    additions=sorted(new.keys()-old)
    changed_paths=(all_sources(SNAPSHOT)|all_sources(SCORE_BASE))&set(fetch.git('diff','--name-only',SCORE_BASE,SNAPSHOT).decode().splitlines())
    references=set();added_ref_count=0
    old_sources=all_sources(SCORE_BASE); new_sources=all_sources(SNAPSHOT)
    for p in sorted(changed_paths):
        a=read_score(p,SCORE_BASE,old_sources);b=read_score(p,SNAPSHOT,new_sources)
        new_ids=b.keys()-a.keys();added_ref_count+=len(new_ids);references.update(new_ids)
    info=dict(start=SCORE_BASE,end=SNAPSHOT,before=old_audit,after=new_audit,
              changed_score_indices=sorted(changed_paths),new_references=added_ref_count,new_reference_ids=len(references),
              newly_seen_score_ids=additions,first_seen_top5=[k for k in additions if set(new[k]['identity'][0])&set(fetch.NAMES)],
              compared_ids_including_mirrors=sorted(references))
    return {k:new[k]['card'] for k in sorted(references)},info

def recent(cache):
    cache.mkdir(exist_ok=True,parents=True)
    cards,info=score_census(); replay_paths=registry.git('ls-tree','-r','--name-only',SNAPSHOT).decode().splitlines()
    names={}
    for p in replay_paths:
        name=p.rsplit('/',1)[-1]
        if name.endswith('.json.gz'): names.setdefault(name,[]).append(p)
    retriever=fetch.Retrieval({},0.2); records=[]
    for gid,card in cards.items():
        local=cache/(gid+'.bin')
        url=fetch.SERVER+f"/replay/{card['id']}/{card['game']}"
        paths=names.get(gid+'.json.gz',[])
        if paths:
            data=registry.git('show',SNAPSHOT+':'+paths[0])
        elif local.is_file():
            data=local.read_bytes()
        else:
            data=retriever.get(url)
            fetch.atomic(local,data)
        env,_,_=fetch.extract(data); checked=fetch.validate(env,card)
        sha=registry.map_identity.terrain_sha256(env['steps'][0][0]['observation']['full_carbon'])
        records.append(dict(game=gid,seed=checked['random_seed'],opening_225_u8_sha256=sha,score_by_seat=checked['rewards_by_seat'],
                            seat_names=checked['seat_model_names'],terminal=checked['statuses'],steps=checked['last_step'],
                            frozen_replay_path=paths[0] if paths else None,official_replay_url=url))
    info['verified_replays']=records
    return info

def freeze(file,value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if file.is_file():
        if file.read_text()!=raw:raise ValueError('Refusing to replace a different receipt: '+str(file))
    else:
        with file.open('x') as dst:dst.write(raw)

def finish(census,inc):
    if census['snapshot']!=inc['end'] or census['seeds']!=SEEDS or census['seed_intersections'] or census['physical_intersections']:
        raise ValueError('Invalid or overlapping repository census')
    layouts=set(census['by_seed'].values()); rows=inc['verified_replays']
    if {r['game'] for r in rows}!=set(inc['compared_ids_including_mirrors']) or len(rows)!=len(inc['compared_ids_including_mirrors']):
        raise ValueError('Incomplete physical-opening responses')
    for r in rows:
        if r['seed'] in SEEDS or r['opening_225_u8_sha256'] in layouts: raise ValueError('Previously played ladder board: '+r['game'])
    # Explicit comparison with the last prospective 26-board registry and previous delta evidence.
    old=registry.git('show',SNAPSHOT+':'+(LEAGUE/'clearance.json').relative_to(registry.ROOT).as_posix())
    prior=json.loads(old)['by_seed']
    earlier=json.loads(registry.git('show',SNAPSHOT+':'+(LEAGUE/'top5_increment.json').relative_to(registry.ROOT).as_posix()))['additional_non_top5_game_openings']
    if set(map(int,prior))&set(SEEDS) or set(prior.values())&layouts or any(r['official_game_seed'] in SEEDS or r['opening_225_u8_sha256'] in layouts for r in earlier):
        raise ValueError('Earlier league or ladder-development registry overlaps')
    # The final delta consists only of the completed twelve-game historical
    # examination, its setup-only attempt, documentation and verifier. Compare
    # the actual registered tickets with the settled rows before advancing the cutoff.
    changed=registry.git('diff','--name-only',SNAPSHOT,FINAL).decode().splitlines()
    root_name=str(LEAGUE.relative_to(registry.ROOT))
    natural_name=root_name+'/natural_original'
    ancillary={root_name+'/README.md',natural_name+'/settle.py'}
    if not changed or any(not(p.startswith(natural_name+'/hosted/') or p in ancillary) for p in changed):
        raise ValueError('Additional unexamined change in the post-census source range')
    touched=all_sources(FINAL)|all_sources(SNAPSHOT)
    if set(changed)&touched:
        raise ValueError('Additional score source requires full score recensus')
    meta=json.loads(registry.git('show',FINAL+':'+natural_name+'/hosted/settlement.json'))
    scored={(r['seed'],r['seat'],r['arm']):r['terrain'] for r in meta['games']}
    if len(scored)!=12 or any(r['status']!='complete' for r in meta['games']) or not meta['records_reconciled']:
        raise ValueError('Historical score settlement incomplete')
    verified={}
    before_start=0
    delta_objects={}
    for path in sorted(changed):
        data=registry.git('show',FINAL+':'+path)
        delta_objects[path]=registry.sha(data)
        if not path.endswith('.zip'):continue
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.testzip() is not None:raise ValueError('Truncated execution archive')
            for name in archive.namelist():
                if not name.endswith('/registration.json'):continue
                stamp=json.loads(archive.read(name));arm=name.split('/')[0];key=(stamp['seed'],stamp['candidate_seat'],arm)
                terrain=stamp['physical_map_sha256']
                if stamp['seed'] in SEEDS or terrain in layouts:raise ValueError('Live original game terrain overlap')
                if '/setup_only/' in path:
                    if arm+'/attempt.json' in archive.namelist():raise ValueError('Historical setup counted as new episode')
                    before_start+=1
                else:
                    if key in verified or scored.get(key)!=terrain:raise ValueError('Unexpected or repeated scored game')
                    verified[key]=terrain
    if verified!=scored or {r[0] for r in verified}!={621758345,621758346}:
        raise ValueError('Twelve-game ticket census mismatch')
    census.update(snapshot=FINAL,source_snapshot=FINAL,source_tree=registry.git('rev-parse',FINAL+'^{tree}').decode().strip(),
                  full_tree_scan_snapshot=SNAPSHOT,increment_through_source_snapshot=dict(files=delta_objects,scored_games=len(verified),
                    scored_by_seed={str(k):sum(g[0]==k for g in verified) for k in sorted({r[0] for r in verified})},
                    setup_registrations=before_start,top5_score_indices_unchanged=True))

    census.update(latest_score_delta=inc,prior_checked_league_boards=len(prior),prior_league_clearance_sha256=hashlib.sha256(old).hexdigest(),
                  old_444_445_reserved={s:prior[str(s)] for s in [621758444,621758445]},other_prior_official_ladder_openings=len(earlier),
                  scope='The source snapshot predates this registration. A passing full scan finds no exact proposed seed or physical layout in the historical tree; code or planning text alone is not evidence of a played match. This report separately verifies the actual openings of recent scored games. A map that is only reserved still stays excluded from this diagnostic.')
    freeze(HERE/'clearance.json',census)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','census','recent','finish'],default='all')
    parser.add_argument('--scratch',required=True,type=Path)
    a=parser.parse_args();a.scratch.mkdir(exist_ok=True,parents=True)
    if a.stage in ('all','census'):freeze(a.scratch/'census.json',full_census())
    if a.stage in ('all','recent'):freeze(a.scratch/'recent.json',recent(a.scratch/'replays'))
    if a.stage in ('all','finish'):finish(json.loads((a.scratch/'census.json').read_bytes()),json.loads((a.scratch/'recent.json').read_bytes()))
if __name__=='__main__':main()

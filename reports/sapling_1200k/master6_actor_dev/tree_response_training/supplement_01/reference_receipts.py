"""Distinguish a byte-verified reservation cross-check from an observed game."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/'reports/sapling_1200k'
OLD='reports/sapling_1200k/all_top5/appendix_97de1ebc/exposure.json'
NEW='reports/sapling_1200k/all_top5/appendix_2cc7c358/exposure.json'
PINNED={OLD:'8e8c1e0de0f8e18f5daee82e7517c1181bde6d5e1bdd5475b5be541d3ce9d6c4',
        NEW:'288db90393cf2c76c3551781821bbd1d88bbe849216c1eb12d14b4127ae38126'}


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prove(name,seed,board,plan,frozen):
    def read(path):
        data=(ROOT/path).read_bytes()
        require(digest(data)==PINNED[path],'changed historical exposure declaration')
        return json.loads(data)
    old=read(OLD);record=old if name==OLD else read(NEW)
    if frozen:
        raw=subprocess.check_output(['git','show',f"{plan['source_revision']}:{name}"],cwd=ROOT)
        require(digest(raw)==PINNED[name],'different historical reference at frozen revision')
    key=plan['intake_name'];references=old['nonoverlap_with_registered_training_and_development'][key]
    require(seed in plan['train_maps'] and references=={'seeds':plan['train_maps'],'seed_hits':[],'terrain_hits':[]},
            'candidate does not match the signed registration')
    sources=list(old['identity_sources'])
    if name==NEW:
        require(record['previous_receipt_sha256']==PINNED[OLD]
                and record['checked_development_and_training_zero_overlap'][key]==plan['train_maps'],'broken exposure receipt chain')
        sources.append(record['new_source'])
        require(record['all_source_sha256']==digest(json.dumps(sources,separators=(',',':')).encode()),'different ledger set')
    seen=set()
    for filename,count,rows_hash,identity_hash in sources:
        raw=gzip.decompress((BASE/filename).read_bytes());require(digest(raw)==rows_hash,'different historical map evidence')
        games=[json.loads(line) for line in raw.splitlines()];ids={g['id'] for g in games}
        require(len(games)==len(ids)==count and not seen.intersection(ids) and digest('\n'.join(sorted(ids)).encode())==identity_hash,
                'incomplete historical map identities')
        require(all(g['seed']!=seed and g['initial_terrain_u8_sha256']!=board for g in games),
                'real opening or nominal seed occurs in the reviewed games')
        seen.update(ids)
    require(len(seen)==record.get('distinct_ids_seeds_and_initial_terrains',record.get('distinct_terrain_for_exclusion')),
            'exposure declaration omits reviewed maps')
    return PINNED[name]


def checked_scan(scan,seed,board,plan):
    """Return unchanged evidence except a proven literal mention inside these two signed receipts."""
    local={str(Path(ROOT.name)/p):p for p in PINNED};accepted={};remaining=[]
    for conflict in scan['conflicts']:
        where,method,*detail=conflict
        if where in local and method=='exact_number_or_terrain' and not detail:
            name=local[where];accepted[name]=prove(name,seed,board,plan,False);continue
        if where=='frozen_git' and method=='unexpected_exact_references' and len(detail)==1:
            nominal=[]
            for name in detail[0]:
                if name in PINNED:accepted[name]=prove(name,seed,board,plan,True)
                else:nominal.append(name)
            if nominal:remaining.append([where,method,nominal])
        else:remaining.append(conflict)
    return {**scan,'conflicts':remaining},accepted

"""Separate a fast observed-map collision check from the full historical number/range audit."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
BASE=ACTOR.parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
import map_identity as mi
from tree_win_gate import RESERVED_RANGE
from tree_audited_game import exam
from official_game_replay import make

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def load(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)

def save(path,obj):
    dest=path.with_suffix('.tmp')
    dest.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    dest.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('early','complete'));ap.add_argument('--work',type=Path,required=True)
    a=ap.parse_args();a.work.mkdir(exist_ok=True,parents=True)
    protocol=HERE/'protocol.json';p=load(protocol);seeds=p['train_maps']+p['development_maps']
    if len(set(seeds))!=4:raise ValueError('expected four different maps')
    signature=sha(protocol.read_bytes());rev=p['source_revision']
    frozen_ladder,known,boards=mi.ladder_boards()
    known=set(known);boards=set(boards)
    accepted={};past={};past_records=set();past_seeds=set();seen_ids=set()
    for patch,*_ in mi.SOURCES:
        for x in gzip.decompress((BASE/patch).read_bytes()).splitlines():seen_ids.add(json.loads(x)['id'])
    extra={}
    for replay in (ROOT.parent/'sapling-all-top5-20261119/replays').glob('*.json.gz'):
        gid=replay.name.split('.')[0]
        if gid in seen_ids:continue
        data=load(replay)
        ob=data['steps'][0][0]['observation']
        if ob['step']!=0:raise ValueError('missing fresh opening')
        terrain=mi.terrain_sha256(ob['full_carbon']);s=data['configuration']['randomSeed']
        extra[gid]={'seed':s,'terrain':terrain}
        known.add(s);boards.add(terrain)
    catalog_roots=(ACTOR/'freight_deployment',ACTOR/'terminal_return_fit')
    evidence=[]
    for base in catalog_roots:
        for file in base.rglob('*'):
            if file.name in ('map_preflight.json','map_clearance.json','clearance.json') or (file.name.startswith(('map_audit_','map_clearance_')) and file.suffix=='.gz'):
                evidence.append(file)
    for file in sorted(evidence):
        old=load(file)
        records=old.get('local_native_or_training_maps',[])
        for x in records:
            past_records.add(x['terrain'])
            if x['seed'] is not None:past_seeds.add(x['seed'])
        named=old.get('boards',{})
        if isinstance(named,dict):
            for name,x in named.items():
                if str(name).isdigit():
                    if isinstance(x,dict):x=x.get('terrain_sha256',x.get('initial_terrain_u8_sha256'))
                    if isinstance(x,str) and len(x)==64:
                        past_records.add(x);past_seeds.add(int(name))
        past[str(file.relative_to(ROOT))]={'sha256':sha(file.read_bytes()),'registry_rows':len(records)}
    for seed in seeds:
        np.random.seed(seed)
        game=make('carbon',configuration={**exam.CFG,'randomSeed':seed})
        board=mi.terrain_sha256(game.states[0].observation.full_carbon)
        if (board!=mi.local_board(seed) or seed in known|past_seeds or board in boards|past_records|set(accepted.values()) or
                RESERVED_RANGE[0]<=seed<RESERVED_RANGE[1]):
            raise ValueError(f'exposed/nonreproducible seed or board: {seed}')
        accepted[str(seed)]=board
    if a.stage=='early':
        out={'stage':'observed_map_and_historical_registry_only','source_revision':rev,'declaration_sha256':signature,
             'boards':accepted,'ladder_games':len(seen_ids),'ladder_hash':frozen_ladder,'additional_cached_public_openings':extra,
             'historical_terrains':len(past_records),'historical_nominal_seeds':len(past_seeds),'prior_catalogs':past}
        save(a.work/'early.json',out);print(json.dumps({k:out[k] for k in ('stage','boards','ladder_games','historical_terrains','historical_nominal_seeds')}),flush=True);return
    first=load(a.work/'early.json')
    if accepted!=first['boards'] or signature!=first['declaration_sha256']:raise ValueError('changed registration or stages')
    audits={}
    for s in seeds:
        raw=(a.work/f'{s}.json').read_bytes();r=json.loads(raw)
        if (r['candidate_seed']!=s or r['candidate_initial_terrain_u8_sha256']!=accepted[str(s)] or
                r['official_opening_sha256']!=accepted[str(s)] or r['frozen_git_revision']!=rev or r['current_declaration']['sha256']!=signature
                or r['conflicts'] or r['same_terrain_matches'] or r['verdict']!='eligible_in_checked_scope'):
            raise ValueError(f'failed historical inventory: {s} {r["conflicts"]} {r["same_terrain_matches"]}')
        archive=HERE/f'clearance_audit_{s}.json.gz'
        archive.write_bytes(gzip.compress(raw,mtime=0))
        audits[str(s)]={'archive':archive.name,'sha256':sha(archive.read_bytes()),'coverage':{
                          k:r['coverage'][k] for k in ('unique_file_or_member_payloads','distinct_nonladder_recorded_terrains','distinct_nonladder_inferred_terrains')},
                          'frozen_ladder':r['frozen_ladder']}
    save(HERE/'clearance.json',{**first,'stage':'observed_maps_and_historical_seed_ranges','verdict':'eligible_in_checked_scope','full_audits':audits})

if __name__=='__main__':
    main()

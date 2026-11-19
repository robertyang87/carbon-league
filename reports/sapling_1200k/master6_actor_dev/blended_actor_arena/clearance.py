"""Join frozen all-workspace scans to current Git, physical training openings and the ladder register."""
import argparse
import gzip
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from shared import ACTOR,BASE,CACHE,HERE,PROTO,PUBLISHED,ROOT,git,published,read,require,save,sha
sys.path.insert(0,str(BASE))
from map_identity import local_board
from tree_win_gate import RESERVED_RANGE

DECL=str(PROTO.relative_to(ROOT))
NOTE='reports/sapling_1200k/known_local_maps/README.md'
NOTE_REV='cf8d6e80f'
SLOT_NOTES=git('show',NOTE_REV+':'+NOTE)
INTRODUCED=[l for l in SLOT_NOTES.splitlines() if ('blended_actor_arena/protocol.json').encode() in l]
require(len(INTRODUCED)==1,'missing historical development declaration line')


def strip_self_reference(name,raw,expected):
    if name==DECL:
        require(raw==PROTO.read_bytes(),'registered development file diverged')
        return b''
    if name==NOTE:
        if INTRODUCED[0] in raw:raw=raw.replace(INTRODUCED[0],b'')
    if name.startswith(str(HERE.relative_to(ROOT))+'/'):
        # This directory is the sole owner of these two diagnostics. Signed scan artifacts only.
        if name in {str((HERE/f'audit_{i}.json.gz').relative_to(ROOT)) for i in (0,1)}|{str((HERE/'clearance.json').relative_to(ROOT))}:
            require(sha(ROOT/name)==sha(raw),'our own receipt diverged')
            return b''
    return raw


def delta_scan(plan,head,boards):
    changed=[p.decode() for p in git('diff','--name-only','--diff-filter=ACMR','-z',plan['source_revision'],head).split(b'\0') if p]
    pattern=re.compile(b'(?<![0-9])(?:'+b'|'.join(str(s).encode() for s in boards)+b')(?![0-9])|'+b'|'.join(b.encode() for b in boards.values()))
    details={}
    def scan(name,data):
        if name.endswith('.tgz') or name.endswith('.tar') or name.endswith('.tar.gz'):
            with tarfile.open(fileobj=io.BytesIO(data)) as ar:
                for m in ar:
                    if m.isfile():scan(name+'!'+m.name,ar.extractfile(m).read())
            return
        b=strip_self_reference(name,data,boards)
        if not b:return
        if name.endswith('.gz'):b=gzip.decompress(b)
        if b'\x00' not in b[:8000]:
            require(not pattern.search(b),'new source mentions reserved development map: '+name)
    for p in changed:
        b=git('show',head+':'+p);scan(p,b);details[p]=sha(b)
    return details


def run(head=None,publish=True):
    plan=read(PROTO);boards={s:local_board(s) for s in plan['new_dev']['seeds']}
    require(sha(PROTO)=='1feb5bc914377ddf3ebf941eb955d41b2c60b0b19ba8b6d3b07427fd656c4fcc' and published(PROTO),'publication')
    require(len(set(boards.values()))==2 and all(not RESERVED_RANGE[0]<=s<RESERVED_RANGE[1] for s in boards),'map aliases or reserved')
    names={};aud=[]
    for i,(s,t) in enumerate(boards.items()):
        src=HERE/f'audit_{i}.json.gz';d=read(src) if src.exists() else read(Path('/tmp/blendedactor-arena-map')/f'{i}.json')
        require(d['official_opening_sha256']==t==d['candidate_initial_terrain_u8_sha256'] and d['candidate_seed']==s and d['frozen_git_revision']==plan['source_revision']
                and not d['reserved'] and not d['ladder_seed_or_terrain_match'] and not d['same_terrain_matches'] and d['current_declaration']['sha256']==sha(PROTO)
                and not d['current_declaration']['frozen_exact_references'],'original opening, identity or historical overlap')
        for name,why in d['conflicts']:
            require(name==ROOT.name+'/'+NOTE and why=='exact_number_or_terrain','unexplained exposure '+name)
            before=git('show',plan['source_revision']+':'+NOTE);now=git('show',NOTE_REV+':'+NOTE)
            require(str(s).encode() not in before and t.encode() not in before and str(s).encode() in INTRODUCED[0] and str(s).encode() not in now.replace(INTRODUCED[0],b''),'mixed declaration and historical exposure')
        require(d['verdict']=='eligible_in_checked_scope' or (d['verdict']=='blocked' and len(d['conflicts'])==1),'other scan failure')
        if not src.exists():save(src,d)
        aud.append({'sha256':sha(src),'raw_scanner_verdict':d['verdict'],'only_additional_reference':d['conflicts'],'files_or_members':d['coverage']['file_or_member_paths']})
    seen=set();known=[]
    known+=list(json.loads((ACTOR/'data_split.json').read_bytes())['games'])
    known+=list(json.loads((ACTOR/'data_loss_split.json').read_bytes())['games'])
    ids={r['id']:(r['seed'],r['terrain_sha256']) for r in known}
    for source in CACHE.glob('*.json'):
        v=read(source)
        if not isinstance(v,dict) or 'id' not in v:continue
        if 'file' not in v:continue
        require(v['id'] in ids,'unbound cached actor map '+v['id']);seen.add(v['id'])
    for s,t in [ids[k] for k in seen]:require(s not in boards and t not in boards.values(),'trained map overlap')
    public={}
    for p in [BASE/'all_top5/validation_maps.jsonl.gz',*sorted((BASE/'all_top5').glob('appendix_*/validation_maps.jsonl.gz'))]:
        for l in gzip.decompress(p.read_bytes()).splitlines():
            r=json.loads(l);s=r['seed'];t=r['initial_terrain_u8_sha256']
            require(s not in boards and t not in boards.values(),'public ladder overlap')
            public.setdefault(s,set()).add(t)
        names[str(p.relative_to(ROOT))]=sha(p)
    tip=head or git('rev-parse','origin/main').decode().strip()
    changes=delta_scan(plan,tip,boards)
    ans={'protocol_sha256':sha(PROTO),'registration_main':PUBLISHED,'historical_scan_revision':plan['source_revision'],'supplementary_tip':tip,
         'source_change_sha256':changes,'physical_boards':boards,'full_pre_registration_workspace_audits':aud,'reported_historical_exposure_count':0,
         'post_registration_mention_explained':{'registration':DECL,'formal_exclusion_note':NOTE,'first_parent_without_declaration':plan['source_revision'],'first_revision_with_exclusion':git('rev-parse',NOTE_REV).decode().strip(),'declaration_line_sha256':sha(INTRODUCED[0])},
         'all_ladder_maps':len(public),'all_ladder_registries':names,'external_cached_training_and_eval_map_ids':len(seen),'known_actor_map_registry':len(ids),'eligible_as_development_only':True}
    if publish:save(HERE/'clearance.json',ans)
    return ans


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--head');o=parser.parse_args();r=run(o.head);print(r['all_ladder_maps'],r['external_cached_training_and_eval_map_ids'],r['eligible_as_development_only'])

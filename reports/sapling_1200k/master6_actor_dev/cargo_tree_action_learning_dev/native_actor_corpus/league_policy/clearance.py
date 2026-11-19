"""Freeze all 26 league identities against the pre-run repository, including archived replays."""
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent
SAP = next(p for p in HERE.parents if (p / 'map_identity.py').is_file())
ROOT = SAP.parents[1]
sys.path[:0] = [str(SAP), str(ROOT / 'code'), str(ROOT / 'code/env_src/carbon_challenge-master')]
import numpy as np
import map_identity
from tree_win_gate import RESERVED_RANGE, exam_exposures
from tree_audited_game import exam
from zerosum_env import make

SNAPSHOT = 'e848e0b5a29e4f81a88587504e28f02ef064db6d'
GROUPS = {'train': list(range(621758420,621758438)), 'development': list(range(621758438,621758444)),
          'original_development_reserve': [621758444,621758445]}
SEEDS = [s for seeds in GROUPS.values() for s in seeds]
STRUCTURED = ('.json','.jsonl','.gz','.tgz','.tar','.zip')
META = ('seed','plan','map','panel','protocol','register','registr','clearance','exposure','declar','split','sources','reserved','roster','terrain')
NUM = re.compile(rb'(?<![.a-zA-Z0-9])\d{8,10}(?![.a-zA-Z0-9])')
CANDIDATE = re.compile(rb'(?<![.\d])6217584(?:2\d|3\d|4[0-5])(?![.\d])')
OPENING = re.compile(rb'"(?:initial_full_carbon|full_carbon)"\s*:\s*(\[[\s0-9.,]+\])')

def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])

def sha(b):
    return hashlib.sha256(b).hexdigest()

def scan():
    physical = {}
    for s in SEEDS:
        np.random.seed(s)
        off = make('carbon',configuration={**exam.CFG,'randomSeed':s})
        q = map_identity.terrain_sha256(off.states[0].observation.full_carbon)
        if q != map_identity.local_board(s): raise ValueError(f'Opening disagreement on {s}')
        physical[s]=q
    if len(set(physical.values())) != len(SEEDS) or any(RESERVED_RANGE[0]<=s<RESERVED_RANGE[1] for s in SEEDS):
        raise ValueError('Missing disjoint seed or terrain')
    hpat=re.compile(b'|'.join(h.encode() for h in physical.values()))
    tree=git('ls-tree','-rz',SNAPSHOT)
    paths=defaultdict(list)
    real_seed=set(); terrain=set(); top5={}; faults=set(); k0={}; counts=Counter()
    manifest=hashlib.sha256()
    for raw in tree.split(b'\0'):
        if not raw: continue
        fields,name=raw.split(b'\t',1); name=name.decode(); counts['repo_paths']+=1
        real_seed.update(map(int,NUM.findall(name.encode())))
        if CANDIDATE.search(name.encode()) or hpat.search(name.encode()):raise ValueError(f'Exposed filename {name}')
        if name.endswith(STRUCTURED):
            oid=fields.split()[2].decode();paths[oid].append(name);manifest.update(raw+b'\0')
    def inspect(path,b):
        basename=path.rsplit('/',1)[-1].lower()
        if basename=='validation_maps.jsonl.gz' and '/all_top5/' in path:
            counts['top5_ledgers']+=1
            for ln in b.splitlines():
                r=json.loads(ln); ident=r['id']
                if ident in top5:raise ValueError(f'Duplicate top5 record {ident}')
                top5[ident]=(r['seed'],r['initial_terrain_u8_sha256'])
                if '/unscored/' in path:faults.add(ident)
        if path.startswith('data/k0_calib/') and 'seed_blocks_' in basename:
            for row in json.loads(b)['blocks']:
                lo,hi=map(int,row['seeds'].split('-'))
                if any(lo<=s<=hi for s in SEEDS):raise ValueError(f'K0 overlap in {path}')
            k0[path]=sha(b)
        if any(w in basename for w in META):
            real_seed.update(map(int,NUM.findall(b)))
        if b'6217584' in b and CANDIDATE.search(b):raise ValueError(f'Existing candidate seed {path}')
        if hpat.search(b):raise ValueError(f'Existing physical terrain {path}')
        cells=OPENING.search(b,0,min(len(b),2**20))
        if cells:
            v=json.loads(cells[1])
            if len(v)==225:
                terrain.add(map_identity.terrain_sha256(v))
    def unpack(path,raw):
        if path.endswith(('.tar','.tgz','.tar.gz')):
            if path.startswith('data/sapling_all_top5_20261119/') or '/all_top5/plant_timing_data/' in path:
                counts['localized_top5_volumes_covered_by_ledgers']+=1;return
            with tarfile.open(fileobj=io.BytesIO(raw)) as z:
                for m in z:
                    if m.isfile() and m.name.endswith(STRUCTURED):unpack(path+'/'+m.name,z.extractfile(m).read())
        elif path.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for n in z.namelist():
                    if n.endswith(STRUCTURED):unpack(path+'/'+n,z.read(n))
        else:
            counts['structured_members']+=1
            inspect(path,gzip.decompress(raw) if path.endswith('.gz') else raw)
    with subprocess.Popen(['git','-C',str(ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)as batch:
        for oid,names in paths.items():
            batch.stdin.write((oid+'\n').encode());batch.stdin.flush()
            header=batch.stdout.readline().split()
            if len(header)!=3 or header[1]!=b'blob':raise ValueError(oid)
            raw=batch.stdout.read(int(header[2]))
            if batch.stdout.read(1)!=b'\n':raise ValueError('Bad Git record')
            for name in names:unpack(name,raw)
        batch.stdin.close()
        if batch.wait():raise RuntimeError('Bad Git reader exit')
    real_seed={s for s in real_seed if 0<=s<2**32}
    _,known_s,known_t=exam_exposures(sorted(real_seed))
    ls={v[0] for v in top5.values()};lh={v[1] for v in top5.values()}
    if not len(ls)==len(lh)==len(top5)==1637 or len(faults)!=1 or len(k0)!=6:raise ValueError('Historical ledger incomplete')
    known_s.update(ls);known_t.update(terrain|lh)
    if known_s.intersection(SEEDS) or known_t.intersection(physical.values()):raise ValueError('Historical map overlap')
    # Git's fixed-pattern backend also scans paths not covered by the structured decoder.
    patterns=b''.join((p+'\n').encode() for s,h in physical.items() for p in (str(s),h))
    search=subprocess.run(['git','-C',str(ROOT),'grep','-I','-l','-F','-f','-',SNAPSHOT,'--'],input=patterns,stdout=subprocess.PIPE)
    if search.returncode not in (0,1):raise RuntimeError(f'Git text search failed: {search.returncode}')
    # A literal byte hit can be a fraction of a logged cash value. An exact
    # terrain digest or a distinct numeric ID remains fatal in every source.
    for blob in search.stdout.decode().splitlines():
        data=git('show',blob)
        if CANDIDATE.search(data) or hpat.search(data):raise ValueError(f'Existing exact identity {blob}')
        counts['text_substring_hits_checked_and_excluded']+=1
    return dict(source_snapshot=SNAPSHOT,source_tree=git('rev-parse',SNAPSHOT+'^{tree}').decode().strip(),
                purpose='Training, held self-owned opponent league and an untouched original-opponent development reservation; none of these 26 terrains can enter the separate formal examination.',
                groups=GROUPS,by_seed={str(s):h for s,h in physical.items()},formal_seed_half_open=RESERVED_RANGE,
                verified_engines=['FastCarbon','zerosum_env official carbon'],top5_unique_openings=len(lh),
                top5_games_scored=len(top5)-len(faults),top5_faults=sorted(faults),k0_seed_block_files=k0,
                source_counts=dict(counts),unique_structured_blobs=len(paths),input_manifest_sha256=manifest.hexdigest(),
                conservative_prior_seed_ids=len(known_s),conservative_prior_physical_layouts=len(known_t),
                previously_recorded_opening_digests=len(terrain),seed_intersections=[],physical_intersections=[])

if __name__=='__main__':
    result=scan(); dest=HERE/'clearance.json';data=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    if dest.is_file() and dest.read_text()!=data:raise RuntimeError('Refusing to replace a different existing registration')
    dest.write_text(data)
    print(f"Verified {len(SEEDS)} distinct unused physical layouts against {result['top5_unique_openings']} top-five openings")

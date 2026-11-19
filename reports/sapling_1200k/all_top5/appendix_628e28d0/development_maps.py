"""Compare the registered two new development boards with all frozen ladder openings."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
BASE=TOP.parent
ROOT=BASE.parents[1]
REGISTER=BASE/'master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/fresh_original/clearance.json'
REVISION='628e28d06b0610f6ceebdac3cc632057766bddd2'
BOARDS={
    '621758400':'43f7ff3a8d1d88feed6f908705acfc3fac4c3aa62275ffb91cd3d2ae669234ed',
    '621758401':'ebf39dc853030dd3989f8bb2fbc3cb491e714db8c0a3c5c8cb7bf74c9366b347'
}

def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    roster=json.loads(git('show',f'{REVISION}:{REGISTER.relative_to(ROOT)}'))
    assert roster['physical_boards']==BOARDS
    paths=[p for p in git('ls-tree','-r','--name-only',REVISION,str(TOP.relative_to(ROOT))).decode().splitlines()
           if p.endswith('/validation_maps.jsonl.gz')]
    files={p:git('show',f'{REVISION}:{p}') for p in paths}
    fresh=HERE/'validation_maps.jsonl.gz'
    files[str(fresh.relative_to(ROOT))]=fresh.read_bytes()
    rows=[json.loads(line) for data in files.values() for line in gzip.decompress(data).splitlines()]
    ids={r['id'] for r in rows}
    seeds={r['seed'] for r in rows}
    maps={r['initial_terrain_u8_sha256'] for r in rows}
    assert len(rows)==len(ids)==len(seeds)==len(maps)==1637
    assert not seeds.intersection(map(int,BOARDS)) and not maps.intersection(BOARDS.values())
    out=dict(ladder_revision=REVISION,ladder_ids_including_one_fault=len(ids),physical_ladder_layouts=len(maps),
             old_and_increment_ledger_sha256={p:sha(b) for p,b in files.items()},
             development_registration=str(REGISTER.relative_to(ROOT)),registration_snapshot=roster['snapshot'],
             checked_development_boards=BOARDS,seen_seed_matches=[],seen_225_cell_layout_matches=[],
             increment_ladder_maps=[json.loads(x) for x in gzip.decompress(fresh.read_bytes()).splitlines()])
    (HERE/'development_maps.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':
    main()

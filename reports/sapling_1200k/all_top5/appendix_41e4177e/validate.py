"""Verify the 41e4177e full board/action delta against nine earlier frozen raw/terrain batches."""
import argparse
import hashlib
from pathlib import Path
from intake import HERE,TOP,REV,PREVIOUS,BATCHES,fetch
from validation_delta import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=fetch.CACHE)
    parent=[(b,hashlib.sha256(fetch.git('show',REV+':'+(TOP/b/'validation_maps.jsonl.gz').relative_to(fetch.ROOT).as_posix())).hexdigest()) for b in BATCHES]
    run(p.parse_args().cache,HERE,REV,PREVIOUS,parent)

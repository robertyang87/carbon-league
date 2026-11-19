"""Independently replay the final four cards, including exact older geography and card exclusion."""
import argparse
import hashlib
import subprocess
from pathlib import Path
from prepare import HERE,BATCHES,PREVIOUS,REVISION
from validation_delta import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path('/workspace/sapling-all-top5-20261119'))
    # Origin card freeze predates publication of its own 4-match validation table. Inspect
    # the new Git freeze that already contains each previous signed register incl. 99919cbf.
    roots=[];root=HERE.parents[3]
    for b in BATCHES:
        f=HERE.parent/b/'validation_maps.jsonl.gz';raw=subprocess.check_output(['git','show',REVISION+':'+f.relative_to(root).as_posix()],cwd=root)
        roots.append((b,hashlib.sha256(raw).hexdigest()))
    run(p.parse_args().cache,HERE,REVISION,PREVIOUS,roots)

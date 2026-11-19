"""Start a recoverable owner. job.py takes the nonblocking single-owner lock."""
import os
from pathlib import Path
import subprocess
import sys
from core import HERE

env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","CARBON_DEBUG":"1","PYTHONDONTWRITEBYTECODE":"1"}
with (HERE/"supervisor.log").open("ab",buffering=0) as log:
    p=subprocess.Popen(["nice","-n","19",sys.executable,"-u",str(HERE/"job.py")],cwd=HERE,env=env,start_new_session=True,
                      stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
print(p.pid,flush=True)

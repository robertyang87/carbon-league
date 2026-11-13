"""Portable paths for local review probes; never writes to the repository."""
import os
from pathlib import Path

def paths():
 explicit=os.environ.get('CARBON_REPO')
 candidates=[Path(explicit).expanduser().resolve()] if explicit else []
 if not explicit:
  for start in (Path.cwd(),Path(__file__).resolve().parent):candidates += [start,*start.parents]
 repo=next((p for p in candidates if (p/'code/certify_pack.py').is_file()),None)
 if repo is None:raise RuntimeError('Set CARBON_REPO to the Carbon League checkout root')
 out=Path(os.environ.get('CARBON_REVIEW_OUT','/tmp/carbon-review-round5/runtime')).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
 return repo,out

def env():
 d=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):d[k]='1'
 return d

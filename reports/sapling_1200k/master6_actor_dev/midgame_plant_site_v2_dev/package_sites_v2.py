"""Materialize the exact public player and check all native actions on its two root edits."""
import argparse
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from study import HERE,V1,PACK,MODEL,sr,plan,code
from fastsim import DEFAULT_CONFIG


def build(p):
    base=PACK/'candidate.py'
    if sr.sha(base)!=plan['source_hash']:raise ValueError("parent identity drift")
    wrap='\n_SOIL_IMPL={}\nexec('+repr(code)+',_SOIL_IMPL)\n'
    wrap+='_SOIL_WEIGHTS='+repr(sr.load(MODEL)['coef'])+'\n_SOIL_DEFAULTS='+repr(DEFAULT_CONFIG)+'\n'
    wrap+='''
_soil_parent=agent
_soil_seats={}
def agent(observation, configuration=None):
    original=_soil_parent(observation, configuration)
    seat=observation["player"];tick=observation["step"]
    if tick==0 or seat not in _soil_seats:
        _soil_seats[seat]={},None
    memory,root=_soil_seats[seat]
    if root and tick>root["t"]+1:
        return original
    cfg=_SOIL_DEFAULTS if configuration is None else configuration
    _SOIL_IMPL["record_floor"](observation,memory,cfg)
    if root:
        return _SOIL_IMPL["pay"](observation,memory,original,cfg,root,_SOIL_WEIGHTS)[0]
    root=_SOIL_IMPL["choose"](observation,memory,original,cfg,_SOIL_WEIGHTS)
    if root:
        _soil_seats[seat]=memory,root
        return dict(original,**{root["worker"]:root["alternative"]})
    return original
'''
    p.write_bytes(base.read_bytes()+wrap.encode())
    if p.stat().st_size>10_000_000:raise ValueError("size")
def verify(path):
    p=runpy.run_path(str(V1/'package_sites.py'))['parity']
    p.__globals__['HERE']=HERE;p(path)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);args=parser.parse_args()
    with TemporaryDirectory() as td:
        path=args.out or Path(td)/'candidate.py';build(path);verify(path)

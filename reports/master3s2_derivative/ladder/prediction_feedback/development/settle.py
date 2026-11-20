"""Validate all eight registered tickets and compare candidate and parent within each map and seat."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import matrix as m

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

audit=load("paired_referee_audit",m.BASE/"settle.py")
views=load("paired_public_views",m.STRATEGY/"native_action_onpolicy_intake/observations.py")
audit.m=m
audit.CONTRASTS=(("calibrated","parent"),)
physical=audit.block

def block(archive,seed,seat,run_id,commit,digest):
    games,errors,events,exposure=physical(archive,seed,seat,run_id,commit,digest)
    with TemporaryDirectory() as td,zipfile.ZipFile(archive) as z:
        folder=Path(td)
        audit.unpack(archive,folder)
        for arm in m.ARMS:
            if (folder/arm/"turns.jsonl").exists():
                archived=z.read(arm+"/observations.jsonl.gz")
                m.require(gzip.decompress(archived)==gzip.decompress(views.recover(folder/arm)),
                          "Public observations disagree for "+arm)
    return games,errors,events,exposure

audit.block=block

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=int,required=True)
    parser.add_argument("--commit",required=True)
    parser.add_argument("--matrix-sha",required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("archives",type=Path,nargs="*")
    args=parser.parse_args()
    result=audit.merge(args.run,args.commit,args.archives,args.out,args.matrix_sha)
    print(json.dumps(result["results"],ensure_ascii=False,indent=2))
    raise SystemExit(int(not result["records_reconciled"]))

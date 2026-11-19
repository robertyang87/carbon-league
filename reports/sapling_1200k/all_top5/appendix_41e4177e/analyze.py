"""Run independent cash, decision-history and map audits for the sealed 16-map score increment."""
import argparse
from pathlib import Path
import subprocess
import sys
from intake import HERE,TOP,BATCHES,fetch,scan


def main(cache):
    m=HERE/'manifest.json';s=scan()
    if s['new_unique_games']!=16 or s['newly_archived_raw']!=16:raise ValueError('missing replay members')
    def run(script,*args): subprocess.run([sys.executable,str(script),*map(str,args)],check=True)
    run(HERE/'validate.py','--cache',cache)
    run(TOP/'appendix_499886f1/cases.py','--manifest',m,'--cache',cache)
    run(TOP/'appendix_0137949d/cash.py','--manifest',m,'--cache',cache)
    index=TOP/'context_appendix_41e4177e_index.tsv.gz';receipt=TOP/'context_appendix_41e4177e_receipt.json'
    run(TOP/'context_catalog.py','--manifest',m,'--output',index)
    run(TOP/'context_audit.py','--index',index,'--validation',HERE/'validation_games.jsonl.gz','--out',receipt,'--jobs','1')
    # Fail closed if a later registered set is accidentally included in this historical receipt.
    run(TOP/'future_economy_check.py','--expect-games',1402,'--replay-cache',cache/'replays','--out',HERE/'future_economy_check.json')
    manifests=[TOP/b/'manifest.json' for b in BATCHES]+[m];flags=[item for f in manifests for item in ('--manifest',f)]
    run(TOP/'master6_pressure.py','--cache',cache,'--prior',TOP/'appendix_99919cbf/master6_pressure.json.gz','--out',HERE/'master6_pressure.json.gz',*flags)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=fetch.CACHE);main(p.parse_args().cache)

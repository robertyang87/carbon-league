"""Verify eight hosted single-game records, including independently reconstructed public views."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import subprocess
import tempfile
import zipfile
import matrix
from observations import recover


def merge(run, commit, protocol, archives):
    plan=matrix.check(protocol)
    req=matrix.require
    signed=subprocess.check_output(['git','-C',str(matrix.ROOT),'show',commit+':'+matrix.PLAN.relative_to(matrix.ROOT).as_posix()])
    req(signed==matrix.PLAN.read_bytes(),'Signed protocol differs at hosted commit')
    meta=matrix.timing.gh(f'runs/{run}')
    req(meta['id']==run and meta['head_sha']==commit and meta['event']=='workflow_dispatch' and meta['head_branch']=='main'
        and meta['status']=='completed' and meta['run_attempt']==1 and meta['path']=='.github/workflows/'+matrix.WORKFLOW
        and meta['head_repository']['full_name']=='robertyang87/carbon-league','Different hosted run')
    listing=matrix.timing.gh(f'runs/{run}/artifacts?per_page=100')
    signed_by_name={v['name']:v for v in listing['artifacts']}
    req(len(signed_by_name)==len(listing['artifacts'])==listing['total_count'],'Missing or duplicate artifact page')
    actual={matrix.sha(p):p for p in archives}
    req(len(actual)==len(archives),'Duplicate archives')
    spec=importlib.util.spec_from_file_location('official_block_audit',matrix.BASE/'settle.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    audit.m=matrix
    games=[];errors=[];proofs=[];consumed=set()
    for ticket in plan['schedule']:
        seed,seat=ticket['seed'],ticket['seat']
        name=f'strategic-dev-{seed}-{seat}-{run}'; vals=signed_by_name.get(name)
        try:
            if not vals:raise ValueError('No signed hosted archive')
            digest=vals['digest'].removeprefix('sha256:');path=actual[digest]
            req(vals['workflow_run']['id']==run and vals['workflow_run']['head_sha']==commit and vals['size_in_bytes']==path.stat().st_size
                and digest not in consumed,'Foreign or duplicated archive')
            consumed.add(digest)
            gs,issues,timeline,exposure=audit.block(path,seed,seat,run,commit,protocol)
            if issues:raise ValueError(str(issues))
            g=gs['collector']
            with tempfile.TemporaryDirectory() as td,zipfile.ZipFile(path) as z:
                folder=Path(td)
                audit.unpack(path,folder)
                req('collector/observations.jsonl.gz' in z.namelist(),'Missing public observation archive')
                archived=z.read('collector/observations.jsonl.gz')
                req(recover(folder/'collector')==archived,'Public observation bytes differ from independent replay')
                g['public_observation_archive_sha256']=hashlib.sha256(archived).hexdigest()
            g.update(archive_id=vals['id'],archive_path=path.name)
            games.append(g)
            proofs.append(dict(seed=seed,seat=seat,id=vals['id'],name=name,bytes=vals['size_in_bytes'],sha256=digest,
                               blob_filename=path.name,host_sequence=timeline,exposure=exposure))
        except (KeyError,ValueError,OSError,IndexError,TypeError,AssertionError,zipfile.BadZipFile) as e:
            errors.append(dict(seed=seed,seat=seat,error=str(e)))
            games.append(audit.missing(seed,seat,'collector',str(e)))
    req(consumed==set(actual),'Unregistered archive provided')
    full=audit.outcomes(games)
    return dict(schema='sapling-physical-policy-training-intake/1',use='on-policy training only',run_id=run,commit=commit,protocol_sha256=protocol,
                run_url=meta['html_url'],workflow=meta['path'],workflow_conclusion=meta['conclusion'],games=games,archives=proofs,
                by_seat={str(s):audit.outcomes([g for g in games if g['seat']==s]) for s in (0,1)},all=full,verification_errors=errors,
                registered_games=len(games),unresolved=full['unresolved'],formal_games=0,development_games=0,records_reconciled=not errors)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=int,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--matrix-sha',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('archives',type=Path,nargs='*')
    a=p.parse_args()
    result=merge(a.run,a.commit,a.matrix_sha,a.archives)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    matrix.play.live.save(a.out,result)
    print(json.dumps(result['all']));raise SystemExit(int(not result['records_reconciled']))

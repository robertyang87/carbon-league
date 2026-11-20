"""Complete one of two independently frozen changed-order responses against intact original search."""
import argparse
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tree_archive',HERE.parent/'verify.py')
verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
pair=verify.pair;base=verify.base


def run(n,cpu):
    assert os.sched_getaffinity(0)=={cpu}
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    contract=(HERE/'protocol.json').read_bytes();identity=pair.sha(contract)
    plan=json.loads(contract);pre=(HERE.parent/'results.json.gz').read_bytes()
    assert pair.sha(pre)==plan['root_study_receipt']=='99feed32618d361960f7e331afc520d08c6298cb4219e5030876deb119037e91'
    assert n in (0,1)
    selected=plan['games'][n];r=json.loads(gzip.decompress(pre))['rows'][selected['ordinal']]
    ticket=r['ticket'];seed,side,t=ticket['seed'],ticket['seat'],ticket['step']
    target=HERE/f'{seed}_{side}_result.json'
    if target.exists():raise ValueError('Completed response already exists')
    pair.HERE=HERE;pair.PROTOCOL=identity;verify.HERE=HERE
    parent_path=verify.FAMILY/'cold_start/candidate.py';original_path=verify.FAMILY.parents[1]/'data/master3s2_source.py'
    assert pair.sha(parent_path.read_bytes())=='7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed'
    canonical=subprocess.check_output(['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=verify.FAMILY)
    assert pair.sha(original_path.read_bytes())==pair.sha(canonical)=='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
    parent=base.load('native_parent',parent_path);exam=base.load('native_examiner',original_path)
    for m in (parent,exam):
        assert m._TEMP==1 and m._SEED is None and m._S_ABORT==.58 and m._RX and m._SEARCH_OK
        m._net()
    fs,cmds=pair.tape(ticket);obs=json.loads(fs[t])['observations'][side]
    assert base.digest(obs)==ticket['public_observation_sha256']
    cfg=base.econ.config();cfg.pop('randomSeed',None)
    hist=pair.historical_root(ticket,side,parent,cfg,fs,cmds)
    current=verify.call(parent,'P',hist,{},obs,cfg,(seed+19073+7*t+side)%2**32)
    snap=pair.snapshot(parent)
    a=next(z for z in r['calls'] if z['mode']=='E')
    p=next(z for z in r['calls'] if z['mode']=='P')
    assert a['action']==selected['arms']['E']['order'] and p['action']==selected['arms']['P']['order']
    tr=a['trace']['plan'];probes=dict(E=dict(a,trace=dict(pivot=tr['worker'])),P=p)
    result=pair.continue_pair(ticket,probes,snap,parent,exam,cfg,fs,cmds,plan['package'],n)
    accounts=verify.score_targets(ticket,result,tr['target'],fs,cmds,side)
    summary=dict(protocol_sha256=identity,seed=seed,seat=side,starting_step=t,cpu=cpu,root_reconstruction_wall_s=current['clocks']['wall'],
                 root_parent_reproduced=current['action']==p['action'],responses=result,tree_economy=accounts,
                 terminal_margin_delta=accounts['new']['margin']-accounts['parent']['margin'])
    pair.save(target,summary)
    print(seed,side,'complete',accounts['parent']['scores'],accounts['new']['scores'],summary['terminal_margin_delta'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',required=True,type=int,choices=(0,1));parser.add_argument('--cpu',required=True,type=int)
    a=parser.parse_args();run(a.index,a.cpu)

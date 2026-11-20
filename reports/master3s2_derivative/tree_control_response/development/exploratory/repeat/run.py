"""Second independent and order-reversed diagnostic, using the first study's two recorded commands."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tree_original',HERE.parents[1]/'verify.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
p=v.pair;ref=v.base


def run(i,cpu):
    assert os.sched_getaffinity(0)=={cpu}
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    raw=(HERE/'protocol.json').read_bytes();plan=json.loads(raw);e=plan['replications'][i]
    first=(HERE.parent/'protocol.json').read_bytes();assert p.sha(first)==plan['parent_protocol_sha256']
    sel=json.loads(first)['games'][i]
    archive=(HERE.parents[1]/'results.json.gz').read_bytes();assert p.sha(archive)==plan['root_study_receipt']
    row=json.loads(gzip.decompress(archive))['rows'][sel['ordinal']];ticket=row['ticket']
    s,t,seed=ticket['seat'],ticket['step'],ticket['seed']
    assert (s,t,seed)==(e['seat'],e['step'],e['seed'])
    tgt=HERE/f'{seed}_{s}_result.json'
    if tgt.exists():raise ValueError('Already recorded')
    p.HERE=HERE;p.PROTOCOL=p.sha(raw);v.HERE=HERE
    orig=v.FAMILY.parents[1]/'data/master3s2_source.py';parent=v.FAMILY/'cold_start/candidate.py'
    canonical=subprocess.check_output(['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=v.FAMILY)
    assert p.sha(orig.read_bytes())==p.sha(canonical)=='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
    assert p.sha(parent.read_bytes())=='7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed'
    parent=ref.load('repeated_parent',parent);exam=ref.load('repeated_examiner',orig)
    for m in (parent,exam):
        assert m._TEMP==1 and m._SEED is None and m._S_ABORT==.58 and m._RX and m._SEARCH_OK
        m._net()
    cfg=ref.econ.config();cfg.pop('randomSeed',None);frames,cmds=p.tape(ticket)
    obs=json.loads(frames[t])['observations'][s];assert ref.digest(obs)==ticket['public_observation_sha256']
    hist=p.historical_root(ticket,s,parent,cfg,frames,cmds)
    cur=v.call(parent,'P',hist,{},obs,cfg,e['parent_private_seed'])
    snap=p.snapshot(parent);ev=next(x for x in row['calls'] if x['mode']=='E');prior=next(x for x in row['calls'] if x['mode']=='P')
    assert prior['action']==sel['arms']['P']['order'] and ev['action']==sel['arms']['E']['order']
    # continue_pair constructs just one simultaneous original root call and clones it into both branches.
    deployed_call=p.call;used=[]
    def opponent_call(m,history,view,config,default_seed,costs,tag):
        assert m is exam and view['player']==1-s and view['step']==t and not used
        used.append(e['examiner_private_seed'])
        return deployed_call(m,history,view,config,e['examiner_private_seed'],costs,tag)
    p.call=opponent_call
    tr=ev['trace']['plan'];probes={'E':dict(ev,trace={'pivot':tr['worker']}),'P':prior}
    responses=p.continue_pair(ticket,probes,snap,parent,exam,cfg,frames,cmds,plan['candidate'],1-i)
    assert used==[e['examiner_private_seed']] and list(responses)==e['arm_order']
    econ=v.score_targets(ticket,responses,tr['target'],frames,cmds,s)
    summary=dict(protocol_sha256=p.PROTOCOL,seed=seed,seat=s,starting_step=t,cpu=cpu,arm_order=list(responses),
                 recalled_parent_private_seed=e['parent_private_seed'],used_examiner_private_seed=used[0],
                 free_root_action_happened_to_equal_frozen_parent=cur['action']==prior['action'],responses=responses,tree_economy=econ,
                 terminal_margin_delta=econ['new']['margin']-econ['parent']['margin'])
    p.save(tgt,summary)
    print(seed,s,[(k,v['scores'],v['margin']) for k,v in econ.items()],summary['terminal_margin_delta'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,choices=(0,1),required=True);ap.add_argument('--cpu',type=int,required=True)
    a=ap.parse_args();run(a.index,a.cpu)

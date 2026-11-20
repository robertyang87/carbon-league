"""Gather terminal outcomes of two comparable V2 actions against repeated original responses."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
ROOT=FAMILY.parents[1]
SOURCE=FAMILY/'ladder/prediction_feedback/package/candidate.py'
OPPONENT=ROOT/'sources/master3s2/master3s2.py'
STEP=289
FEATURES=('cash_margin','cargo_margin','near_cashable_margin','tree_count_margin')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def encode(obj):
    return (json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()


def write(path,obj):
    raw=encode(obj)
    if path.suffix=='.gz':
        raw=gzip.compress(raw,mtime=0)
    temp=path.with_suffix('.partial')
    temp.write_bytes(raw);temp.replace(path)
    return sha(raw)


def register():
    p=FAMILY/'final_turn_responses/protocol.json'
    old=json.loads(p.read_bytes())
    assert sha(SOURCE.read_bytes())==old['packages'][0] and sha(OPPONENT.read_bytes())==old['examiner']
    cases=[]
    done=set()
    for t in sorted(old['tickets'],key=lambda r:(r['seed'],r['seat'])):
        arc=FAMILY/t['archive']
        assert sha(arc.read_bytes())==t['archive_sha256']
        with zipfile.ZipFile(arc) as z:
            row=json.loads(gzip.decompress(z.read('parent/observations.jsonl.gz')).splitlines()[STEP])
        t=dict(t,step=STEP,own_public_sha256=row['public_observation_sha256'][t['seat']],
               split='fit' if t['seed']<621782470 else 'held')
        t['rng']=int(sha(f"terminal-paired/1/{t['seed']}/{t['seat']}/root".encode())[:8],16)
        t['rival_rng']=t['rng']^0x7D4A1F6B
        assert (t['seed'],t['seat']) not in done
        done.add((t['seed'],t['seat']))
        cases.append(t)
    assert len(cases)==16
    return dict(schema='v2-signed-terminal-paired-return/1', initial=sha(p.read_bytes()),
                parent=old['packages'][0],examiner=old['examiner'],features=FEATURES,
                tickets=cases, work=dict(cap=8, depth=8, samples=1, repeats=2),
                selection='Fixed t289 on all eight archived maps and both seats. Equal offline H8 worker budget. '
                          'Choose the best two complete candidates without labels. Re-run two native original streams per root. '
                          'Split by earlier/later source panel, never use formal exam maps.')


def board_features(state, me):
    def account(seat):
        cash,bases,workers,trees=state['players'][seat]
        homes=list(bases.values())
        cargo=0.
        near=0.
        for pos,value,role in workers.values():
            if role != 'COLLECTOR':
                continue
            cargo+=value
            row,col=divmod(pos,15)
            dist=min((min(abs(row-b//15),15-abs(row-b//15))
                     +min(abs(col-b%15),15-abs(col-b%15)) for b in homes),default=600)
            if dist <= 299-state['step']:
                near+=value
        return [cash,cargo,near,len(trees)]
    return [a-b for a,b in zip(account(me),account(1-me),strict=True)]


def run(p,cases,cpu,out):
    os.sched_setaffinity(0,{cpu})
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path[:0]=[str(ROOT/'code/env_src/carbon_challenge-master'),
                 str(FAMILY/'symmetry_value/terminal_pairs'),
                 str(FAMILY/'opponent_policy/search_probe')]
    import branches
    from full_search_check import root_call
    loader=load('terminal_pair_native_loader',FAMILY/'single_worker_priority/response/run.py')
    signed=loader.load_runner()
    signed.PLAN=p
    f=branches.freight
    live=f.load('pair_official',f.STUDY/'cargo_tree_action_learning_dev/ladder_live/live.py')
    branches.advance=signed.checked_step
    original=SOURCE.read_bytes()
    assert sha(original)==p['parent'] and sha(OPPONENT.read_bytes())==p['examiner']
    raw=original.decode().replace('\r\n','\n')
    marker='        depth = t\n        if t == D2 or not n:'
    assert raw.count(marker)==1
    changed=raw.replace(marker,'''        depth = t
        if t == D2:
            _PAIR_LEAVES.update({b[0]: _PAIR_EXTRACTOR(b[2], me) for b in run})
        if t == D2 or not n:''')
    # Constrain only the research-side work to the predeclared menu cap and depth. No edited module is played in continuations.
    mod=type(sys)('offline_terminal_labels')
    exec(compile(changed,'terminal-pair-research','exec'),mod.__dict__)
    mod._PAIR_EXTRACTOR=board_features
    mod._S_K=8
    mod._S_ABORT=30.
    mod._net()
    torch.set_num_threads(1)
    all_rows=[]
    for t in cases:
        seed,seat=t['seed'],t['seat']
        paths=[SOURCE if i==seat else OPPONENT for i in (0,1)]
        bots=[f.load(f'original_continuation_{i}',q) for i,q in enumerate(paths)]
        for m in bots:
            assert m._net() is not None and m._TEMP==1 and m._S_ABORT==.58 and m._SEED is None
            m._torch.set_num_threads(1)
        model_state=signed.replay_root(t,bots,f,live)
        _,cfg,_,_,obs,restored=model_state
        parent=bots[seat]
        mod._PAIR_LEAVES={}
        result=root_call(mod,parent._ST['tr'],parent._ST['prev'],obs[seat],cfg,t['rng'],t['initial_cost'])
        eligible=[i for i,d in enumerate(result['depth_by_arm']) if d==8]
        evidence=dict(ticket=t, roots=result, leaves=mod._PAIR_LEAVES, branches=[], own_public=restored['public_sha256_by_seat'])
        if len(eligible)<2 or result['best'] not in eligible:
            all_rows.append(evidence)
            print(seed,seat,'ineligible',eligible,flush=True)
            continue
        first=result['best']
        second=max((j for j in eligible if j!=first),key=lambda j:result['leaf_scores'][j])
        a,b=result['menu'][first],result['menu'][second]
        assert a!=b and first in mod._PAIR_LEAVES and second in mod._PAIR_LEAVES
        evidence['choices']={'first':first,'second':second}
        for stream in range(2):
            salt=int(sha(f'terminal-pair-stream/1/{seed}/{seat}/{stream}'.encode())[:8],16)
            ticket=t|dict(rng=t['rng']^salt,rival_rng=t['rival_rng']^salt)
            state=signed.replay_root(ticket,bots,f,live)
            _,cfg,_,_,obs,incoming=state
            init=f.original_state.h([m._ST for m in bots])
            rng=f.original_state.h(parent._ST['rng'])
            initial=f.measured(parent,obs[seat],cfg)
            tags=['first','second'] if (seed+seat+stream)%2==0 else ['second','first']
            def save(name,value):
                return write(out/f'{stream}_{name}',value)
            paired=branches.continue_pair(ticket,{'first':a,'second':b},tags,state,initial,init,rng,save)
            assert all(g['statuses']==['DONE','DONE'] for g in paired['arms'].values())
            paired['stream']=stream
            evidence['branches'].append(paired)
        all_rows.append(evidence)
        print(seed,seat,[round(r['arms']['first']['margin']-r['arms']['second']['margin'],2) for r in evidence['branches']],flush=True)
    write(out/'rows.json.gz',dict(protocol=sha(encode(p)),rows=all_rows))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    cli.add_argument('--part',type=int)
    a=cli.parse_args()
    p=register()
    k=HERE/'protocol.json'
    if a.register:
        assert not k.exists() or k.read_bytes()==encode(p)
        print(write(k,p))
    else:
        assert k.read_bytes()==encode(p) and 0<=a.part<4
        out=HERE/f'part{a.part}'
        out.mkdir(exist_ok=True)
        run(p,p['tickets'][a.part::4],a.cpu,out)

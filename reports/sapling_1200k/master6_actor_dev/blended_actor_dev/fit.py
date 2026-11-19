"""Learn and evaluate an actor on the scores used jointly by the two deployment nets."""
import ast
import base64
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import zlib

import numpy as np
import torch
from torch.nn.functional import cross_entropy

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
spec = importlib.util.spec_from_file_location('signed_actor_trainer', ACTOR / 'train.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
PROTOCOL = json.loads((HERE / 'protocol.json').read_bytes())
PARENT = ROOT / 'data/weights/sapling_1200k/master6_actor_v1.npz'
DIR = Path('/workspace/sapling-master6-actor-data')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expert_model():
    path = ACTOR / 'actor_partner_dev/candidate.py'
    raw = path.read_bytes()
    if common.digest(raw) != PROTOCOL['partner_pack_sha256']:
        raise ValueError('frozen partner changed')
    source = ast.parse(raw)
    fields = {stmt.targets[0].id: ast.literal_eval(stmt.value) for stmt in source.body
              if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
              and stmt.targets[0].id in ('_B1','_M1','_Q1')}
    fn = next(s for s in source.body if isinstance(s, ast.FunctionDef) and s.name == '_dequantize')
    env = {'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]), str(path), 'exec'),env)
    weights = env['_dequantize'](zlib.decompress(base64.b64decode(fields['_B1'])),fields['_Q1'])
    meta = json.loads(fields['_M1'])
    if meta['it'] != 20:
        raise ValueError('wrong expert iteration')
    net = common.CNet2(**meta['cfg'])
    missing, unexpected = net.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()},strict=False)
    if unexpected or any(k.split('.')[0] not in ('wdl','utility','margin') and not k.endswith('num_batches_tracked') for k in missing):
        raise ValueError((missing,unexpected))
    return net.eval().requires_grad_(False)


def load(rows, directory=DIR):
    pieces = []
    for g in rows:
        raw = (directory / (g['id'] + '.npz')).read_bytes()
        if common.digest(raw) != g['shard_sha256']:
            raise ValueError('shard hash: '+g['id'])
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            d = {k:z[k] for k in common.trainer.FIELDS if k not in ('carbon','value')}
        n = len(d['gid'])
        for k in ('carbon','value'):
            dtype,shape = common.trainer.FIELDS[k]
            d[k] = np.zeros((n,*shape),dtype=dtype)
        if n != g['rows'] or (d['gid'] != g['gid']).any() or (d['teacher']!=2).any() or len(np.unique(d['step']))!=n:
            raise ValueError('invalid identity or roots')
        active = d['y'] >= 0
        if not np.array_equal(active,d['tok_mask'][:,:active.shape[1]]) or not d['legal'][np.where(active)+(d['y'][active],)].all():
            raise ValueError('invalid demonstration prefix')
        pieces.append(d)
    return common.join_parts(pieces)


def blended(expert,student):
    return 0.75*expert + 0.25*student


def evaluate(data, records, models, expert, train_opponents):
    probs = {k:[] for k in models}
    for start in range(0,len(data['gid']),16):
        b = common.batch(data,slice(start,start+16))
        with torch.inference_mode():
            e = expert(b)[0]
            for name,model in models.items():
                z = blended(e,model(b)[0])
                if not torch.isfinite(z).all():
                    raise ValueError('nonfinite held scores')
                probs[name].append(z.log_softmax(-1).numpy())
    logp = {k:np.concatenate(p) for k,p in probs.items()}
    y = data['y']
    active = common.actionable(data)
    kinds = data['tok_kind'][:,:y.shape[1]]
    cargo = data['tok_cargo'][:,:y.shape[1]]
    gids = data['gid']
    def summary(mask):
        mask = active & mask
        per_map = []
        for gid in np.unique(gids[mask.any(1)]):
            ok = mask & (gids==gid)[:,None]
            true = y[ok]
            ii = np.arange(len(true))
            pred = {k:lp[ok].argmax(-1) for k,lp in logp.items()}
            per_map.append({'id':records[int(gid)]['id'],'slots':len(true),
                'ce':{k:float(-lp[ok][ii,true].mean(dtype=np.float64)) for k,lp in logp.items()},
                'agreement':{k:float((p==true).mean()) for k,p in pred.items()},
                'changed_vs_parent':{k:int((p!=pred['parent']).sum()) for k,p in pred.items()},
                'corrected_vs_parent':{k:int(((p==true)&(pred['parent']!=true)).sum()) for k,p in pred.items()},
                'damaged_vs_parent':{k:int(((p!=true)&(pred['parent']==true)).sum()) for k,p in pred.items()}})
        out={'maps':len(per_map),'slots':int(mask.sum())}
        if per_map:
            for field in ('ce','agreement'):
                out[field]={k:float(np.mean([g[field][k] for g in per_map])) for k in models}
            for field in ('changed_vs_parent','corrected_vs_parent','damaged_vs_parent'):
                out[field]={k:sum(g[field][k] for g in per_map) for k in models}
        return out,per_map
    filters={r:kinds==i for i,r in enumerate(common.ROLE)}
    leg = data['legal']
    mf = data['mf']
    collector = (kinds==0)
    filters.update({
        'loaded_at_least_40_home_reachable':collector & (cargo>=40) & ((mf[...,5]>0)&leg).any(-1),
        'loaded_at_least_40_lighter_or_equal_foe_reachable':collector & (cargo>=40) & ((mf[...,1]>0)&leg).any(-1),
        'heavier_foe_reachable':collector & ((mf[...,2]>0)&leg).any(-1),
        'opponent_not_in_training':np.array([records[int(g)]['opponent'] not in train_opponents for g in gids])[:,None]})
    for field in ('seat','wdl','opponent'):
        vals=np.array([str(records[int(g)][field]) for g in gids])
        for val in sorted(set(vals)):
            filters[f'{field}={val}']=(vals==val)[:,None]
    all_scores,by_map = summary(active)
    return {'all':all_scores,'by_map':by_map,'subgroups':{k:summary(mask)[0] for k,mask in filters.items()}}


def run():
    torch.set_num_threads(1)
    torch.manual_seed(PROTOCOL['random_seed'])
    if os.sched_getaffinity(0)!={2} or os.getpriority(os.PRIO_PROCESS,0)<19:
        raise ValueError('use one low-priority CPU2')
    panel = HERE / 'panel.json'
    if sha(panel)!=PROTOCOL['panel_sha256']:
        raise ValueError('panel changed')
    games = json.loads(panel.read_bytes())['games']
    train_games=[g for g in games if g['split']=='train']
    train = load(train_games)
    print('loaded train',len(train['gid']),flush=True)
    parent,meta = common.model(PARENT,PROTOCOL['parent_weights_sha256'])
    parent.eval()
    expert = expert_model()
    nets = {k:copy.deepcopy(parent) for k in ('candidate','control')}
    frozen_layers=set(PROTOCOL['train_modules'])
    optim = {}
    for name,model in nets.items():
        for k,p in model.named_parameters():
            p.requires_grad_(k.split('.')[0] in frozen_layers)
        optim[name]=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=PROTOCOL['lr'],weight_decay=PROTOCOL['weight_decay'])
    size = PROTOCOL['batch_size']
    perm = np.random.default_rng(PROTOCOL['random_seed']).permutation(len(train['gid']))[:size*PROTOCOL['updates']].reshape(-1,size)
    losses=[]
    for n,idx in enumerate(perm):
        b=common.batch(train,idx)
        active=common.actionable(b)
        with torch.no_grad():
            e = expert(b)[0]
            tok,ctx,_=parent.encode(b)
        log = {}
        for name,model in nets.items():
            z=model.decode(b,tok,ctx)
            if name=='candidate':
                z=blended(e,z)
            loss=cross_entropy(z[active],b['y'][active])
            optim[name].zero_grad(set_to_none=True)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),PROTOCOL['clip_grad_norm'],error_if_nonfinite=True)
            optim[name].step()
            log[name]=[float(loss.item()),float(norm.item())]
        losses.append(log)
        if (n+1)%16==0:
            print('update',n+1,log,flush=True)
    for name,model in nets.items():
        output = HERE/(name+'.npz')
        tmp = HERE/(name+'.tmp.npz')
        common.trainer.export(model,str(tmp),dict(meta,blended_training_arm=name,protocol_sha256=sha(HERE/'protocol.json')))
        os.replace(tmp,output)
    train_opponents={g['opponent'] for g in train_games if g['rows']>0}
    models={n:common.quantized(m).eval() for n,m in dict(parent=parent,**nets).items()}
    result={'protocol_sha256':sha(HERE/'protocol.json'),'source_sha256':sha(Path(__file__)),'weights':{k:sha(HERE/(k+'.npz')) for k in nets},
            'train_roots':len(train['gid']),'sample_schedule_gid_step_sha256':common.digest(np.stack([train['gid'][perm],train['step'][perm]],axis=-1).astype('<i8').tobytes()),
            'unique_trained_maps':int(len(np.unique(train['gid'][perm]))),'updates':len(perm),'loss_and_unclipped_grad_norm':losses,
            'trainable_parameter_count':sum(p.numel() for p in nets['candidate'].parameters() if p.requires_grad),'scores':{}}
    del train
    for split in ('val','holdout','diagnostic_seen_loss'):
        if split=='diagnostic_seen_loss':
            items=json.loads((ACTOR/'data_loss_split.json').read_bytes())['games']
            receipt=ACTOR/'data_loss_receipts.json.gz'
            assert sha(receipt)==PROTOCOL['diagnostic_previously_exposed_losses_receipt']
            by_id={g['id']:g for g in json.loads(gzip.open(receipt,'rb').read())['games']}
            rows=[dict(g,shard_sha256=by_id[g['id']]['sha256'],rows=by_id[g['id']]['rows']) for g in items]
            directory=Path('/workspace/sapling-master6-actor-exposed-loss')
        else:
            rows=[g for g in games if g['split']==split]
            directory=DIR
        data=load(rows,directory)
        result['scores'][split]=evaluate(data,{g['gid']:g for g in rows},models,expert,train_opponents)
        print(split,result['scores'][split]['all'],flush=True)
    common.write_json(HERE/'fit.json',result)


if __name__=='__main__':
    run()

"""Freeze the final update of the registered Master6 actor and evaluate recorded public prefixes."""
import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code'))
os.environ.setdefault('BC_DEVICE', 'cpu')
import cnet2_train as trainer
import cnet2_pack as pack
from cnet2_model import CNet2
from init_sapling import resolve_preset_metadata
from sapling_soft_it50 import divergence

WEIGHTS = ROOT / 'data/weights/sapling_1200k'
ROLE = ('collector', 'planter', 'base')
POLICY = HERE / 'training_protocol.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def model(path, expected=None):
    raw = path.read_bytes()
    if expected and digest(raw) != expected:
        raise ValueError('weight identity mismatch')
    with np.load(io.BytesIO(raw), allow_pickle=False) as a:
        meta = json.loads(a['meta'].item())
        preset, cfg, total = resolve_preset_metadata(meta)
        if preset != 'sapling-1.2m' or total != 1206476:
            raise ValueError('architecture mismatch')
        net = CNet2(**cfg)
        net.load_state_dict({k: torch.from_numpy(a[k]) for k in a.files if k != 'meta'}, strict=True)
    return net, meta


def actionable(value):
    return (value['y'] >= 0) & (value['legal'].sum(-1) >= 2)


def join_parts(parts):
    return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}


def public(part, manifest, directory, receipts):
    selected = [g for g in manifest if g['split'] == part]
    if not selected:
        raise ValueError(f'empty split {part}')
    arrays = []
    hashes = {}
    for g in selected:
        path = directory / (g['id'] + '.npz')
        raw = path.read_bytes()
        hashes[g['id']] = digest(raw)
        if receipts and receipts.get(g['id']) != hashes[g['id']]:
            raise ValueError('root shard digest changed ' + g['id'])
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            d = {k: z[k] for k in trainer.FIELDS}
        if (d['gid'] != g['gid']).any() or (d['teacher'] != 2).any():
            raise ValueError('root game or style identity changed')
        if len(np.unique(d['step'])) != len(d['step']):
            raise ValueError('repeated public root')
        active = d['y'] >= 0
        if not np.array_equal(active, d['tok_mask'][:, :active.shape[1]]) or not d['legal'][np.where(active) + (d['y'][active],)].all():
            raise ValueError('illegal public target')
        arrays.append(d)
    return join_parts(arrays), hashes


def auxiliary(directory, protocol, public_seeds):
    raw = (directory / 'manifest.json').read_bytes()
    if digest(raw) != protocol['soft_manifest_sha256']:
        raise ValueError('external soft source manifest')
    games = json.loads(raw)['games']
    train = [g for g in games if g['split'] == 'train']
    if {g['seed'] for g in train} != set(range(620176000, 620176018)) or {g['seed'] for g in games} & public_seeds:
        raise ValueError('old and new maps overlap or teacher generation changed')
    pieces = []
    for g in train:
        raw = (directory / g['shard']).read_bytes()
        if digest(raw) != g['sha256']:
            raise ValueError('external soft shard hash')
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            d = {k: z[k] for k in trainer.LOAD}
            q = z['teacher_probs']
            if not np.allclose(q.sum(-1), d['y'] >= 0, atol=1e-5, rtol=0) or np.any(q[~d['legal']]) or not np.isfinite(q).all() or np.any(q < 0):
                raise ValueError('invalid external probability')
            d['q'] = q
        pieces.append(d)
    return join_parts(pieces)


def batch(data, idx):
    return trainer.to_dev({k: data[k][idx] for k in trainer.LOAD}, aug=False)


@torch.inference_mode()
def predict(net, data):
    net.eval()
    q = []
    for i in range(0, len(data['y']), 32):
        z = net(batch(data, slice(i, i + 32)))[0]
        if not torch.isfinite(z).all():
            raise ValueError('nonfinite logits')
        q.append(z.softmax(-1).numpy())
    return np.concatenate(q)


def score(data, a, b, records):
    active = actionable(data)
    y = data['y']
    kind = data['tok_kind'][:, :y.shape[1]]
    def category(field):
        return np.asarray([str(records[int(g)][field]) for g in data['gid']])
    def summarise(selected):
        mask = active & selected
        n = int(mask.sum())
        if n == 0:
            return {'slots': 0}
        truth = y[mask]
        p, q = a[mask], b[mask]
        pa, pb = p.argmax(-1), q.argmax(-1)
        ia, ib = pa == truth, pb == truth
        ii = np.arange(n)
        return {'maps': int(len(np.unique(data['gid'][mask.any(1)]))), 'slots': n,
                'parent_ce': float(-np.log(np.maximum(p[ii, truth], 1e-30)).mean(dtype=np.float64)),
                'candidate_ce': float(-np.log(np.maximum(q[ii, truth], 1e-30)).mean(dtype=np.float64)),
                'parent_correct': int(ia.sum()), 'candidate_correct': int(ib.sum()),
                'parent_agreement': float(ia.mean()), 'candidate_agreement': float(ib.mean()),
                'changed': int((pa != pb).sum()), 'corrected': int((~ia & ib).sum()), 'damaged': int((ia & ~ib).sum()),
                'target_action_counts': np.bincount(truth, minlength=5).tolist(), 'candidate_counts': np.bincount(pb, minlength=5).tolist()}
    result = {'all': summarise(active), 'role': {r: summarise(kind == i) for i, r in enumerate(ROLE)}}
    for f in ('cohort', 'wdl', 'seat'):
        labels = category(f)
        result[f] = {v: summarise((labels == v)[:, None]) for v in sorted(set(labels))}
    result['by_map'] = {records[int(g)]['id']: summarise((data['gid'] == g)[:, None]) for g in np.unique(data['gid'])}
    return result


def quantized(net):
    out = copy.deepcopy(net)
    _, _, weights = pack.quantize({k: v.detach().numpy() for k, v in net.state_dict().items()})
    out.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=True)
    return out


def run(a):
    if 'BC_DEVICE' in os.environ and os.environ['BC_DEVICE'] != 'cpu' or pack.QMODE != 'int8':
        raise ValueError('CPU and int8 required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads(POLICY.read_bytes())
    rng = np.random.default_rng(protocol['seed'])
    torch.manual_seed(protocol['seed'])
    split_raw = a.split.read_bytes()
    manifest = json.loads(split_raw)['games']
    usable = [g for g in manifest if g['split'] in ('train', 'val', 'holdout')]
    records = {int(g['gid']): g for g in usable}
    if len(records) != len(usable) or len({g['seed'] for g in usable}) != len(usable) or len({g['terrain_sha256'] for g in usable}) != len(usable):
        raise ValueError('duplicate game/map identity')
    receipts = {}
    if a.receipts:
        with gzip.open(a.receipts, 'rt') as source:
            receipt_doc = json.load(source)
        if receipt_doc['split_sha256'] != digest(split_raw):
            raise ValueError('receipts bind another selection')
        for g in receipt_doc['games']:
            if 'sha256' in g:
                receipts[g['id']] = g['sha256']
        if len(receipts) != len(usable):
            raise ValueError('incomplete data receipts')
    before_time = time.monotonic()
    parent, meta = model(WEIGHTS / 'cash_v3.npz', protocol['parent_sha256'])
    net = copy.deepcopy(parent)
    train, identities = public('train', usable, a.data, receipts)
    soft = auxiliary(a.soft, protocol, {g['seed'] for g in usable})
    free = actionable(train)
    kinds = train['tok_kind'][:, :free.shape[1]]
    rows = np.flatnonzero(free.any(1))
    per_role = np.asarray([(free & (kinds == i)).sum() for i in range(3)])
    if (per_role == 0).any():
        raise ValueError('absent required role')
    scale = torch.tensor(np.array([.52, .27, .21]) * len(rows) / per_role, dtype=torch.float32)
    params = [[], []]
    trunk, learn = tuple(protocol['learn'][:2]), tuple(protocol['learn'])
    for k, p in net.named_parameters():
        p.requires_grad_(k.startswith(learn))
        if p.requires_grad:
            params[0 if k.startswith(trunk) else 1].append(p)
    opt = torch.optim.AdamW([{'params': params[0], 'lr': protocol['lr_encoder']},
                             {'params': params[1], 'lr': protocol['lr_decoder']}], weight_decay=.0001)
    order = rng.permutation(rows)
    # Cache these distributions once. The teacher sees the same actual legal prefixes.
    prior = predict(parent, train)
    soft_free = actionable(soft)
    soft_rows = np.flatnonzero(soft_free.any(1))
    history = []
    net.train()
    for offset in range(0, len(order), 32):
        ix = order[offset:offset + 32]
        si = rng.choice(soft_rows, 4, replace=False)
        joined = {k: np.concatenate((train[k][ix], soft[k][si])) for k in trainer.LOAD}
        b = trainer.to_dev(joined, aug=False)
        z = net(b)[0]
        y = b['y'][:len(ix)]
        mask = torch.as_tensor(free[ix])
        role = torch.as_tensor(kinds[ix], dtype=torch.int64)
        w = mask * scale[role.clamp(max=2)]
        ce = (F.cross_entropy(z[:len(ix)].transpose(1,2), y, ignore_index=-1, reduction='none') * w).sum() / len(ix)
        anchor = (divergence(z[:len(ix)], torch.from_numpy(prior[ix])) * w).sum() / len(ix)
        teach = divergence(z[len(ix):], torch.from_numpy(soft['q'][si]))[torch.as_tensor(soft_free[si])].mean()
        loss = .80 * ce + .12 * anchor + .08 * teach
        if not torch.isfinite(loss):
            raise ValueError('nonfinite objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 2., error_if_nonfinite=True)
        opt.step()
        if (offset//32) % 20 == 0 or offset + 32 >= len(order):
            history.append(dict(update=offset//32+1, hard=float(ce.detach()), self_kl=float(anchor.detach()), it50_kl=float(teach.detach()), norm=float(norm)))
            print(history[-1], flush=True)
    changed = {}
    for name, p in net.named_parameters():
        delta = (p - dict(parent.named_parameters())[name]).detach()
        if torch.count_nonzero(delta):
            if not p.requires_grad:
                raise ValueError('changed frozen tensor')
            changed[name] = {'elements_changed': int(torch.count_nonzero(delta)), 'elements': p.numel(), 'max_abs': float(delta.abs().max())}
    if not {key.split('.')[0] for key in changed} >= {'blocks','attn','gru','bhead','whead'}:
        raise RuntimeError('missing genuine multi-layer update')
    recipe = dict(protocol_sha256=digest(POLICY.read_bytes()), split_sha256=digest(split_raw),
                  code_sha256=digest(Path(__file__).read_bytes()), train_source_shards=identities,
                  parent_sha256=protocol['parent_sha256'], selected='fixed_final_update',
                  total_updates=(len(order)+31)//32, roots=len(rows), perm_sha256=digest(order.astype('<i8').tobytes()),
                  train_maps=len(np.unique(train['gid'])), train_slots=per_role.tolist(), optimizable=sum(p.numel() for ps in params for p in ps),
                  elapsed_to_freeze_seconds=time.monotonic()-before_time)
    meta = dict(meta, training_status='master6_public_episode_actor', master6_actor={k:v for k,v in recipe.items() if k not in ('elapsed_to_freeze_seconds','train_source_shards')})
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        w, agent = Path(tmp)/'model.npz', Path(tmp)/'agent.py'
        trainer.export(net, str(w), meta)
        pack.main(str(w), str(agent), 2)
        os.replace(w, a.out)
        os.replace(agent, a.pack)
    packet=dict(npz=a.out.name, agent=a.pack.name, npz_sha256=digest(a.out.read_bytes()), agent_sha256=digest(a.pack.read_bytes()), recipe=recipe, history=history, changed=changed)
    write_json(a.report_dir/'weight_receipt.json',packet)
    print('FROZEN',packet['npz_sha256'],packet['agent_sha256'],flush=True)
    quant_parent, quant_net = quantized(parent), quantized(net)
    result = {}
    for part in ('val','holdout'):
        d, hashes = public(part, usable, a.data, receipts)
        result[part] = {'roots': len(d['y']), 'shards': hashes, 'fp32': score(d,predict(parent,d),predict(net,d),records),
                        'int8': score(d,predict(quant_parent,d),predict(quant_net,d),records)}
        print(part, result[part]['int8']['all'], flush=True)
    write_json(a.report_dir/'metrics.json',dict(weights={k:v for k,v in packet.items() if k.endswith('sha256')}, results=result,
                elapsed_seconds=time.monotonic()-before_time, torch=str(torch.__version__), numpy=np.__version__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,default=HERE/'data_split.json')
    p.add_argument('--receipts',type=Path)
    p.add_argument('--report-dir',type=Path,default=HERE)
    p.add_argument('--data',type=Path,default=Path('/workspace/sapling-master6-actor-data'))
    p.add_argument('--soft',type=Path,default=Path('/workspace/sapling-soft-it50-data'))
    p.add_argument('--out',type=Path,default=WEIGHTS/'master6_actor_v1.npz')
    p.add_argument('--pack',type=Path,default=WEIGHTS/'master6_actor_v1.py')
    args=p.parse_args()
    if args.out.exists() or args.pack.exists():
        raise FileExistsError('final candidate already exists')
    run(args)

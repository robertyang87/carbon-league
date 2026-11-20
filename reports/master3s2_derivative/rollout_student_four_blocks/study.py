"""Fit a regularized block-four rollout readout using archived original search boards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ARCHIVE = FAMILY / 'rollout_policy_learning/student'
sys.path.insert(0, str(ARCHIVE))

SOURCE_SHA = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
ROOT = FAMILY.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--cpu', type=int, required=True)
    cli.add_argument('phase', choices=('register', 'fit', 'held'))
    args = cli.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F

    import learn
    from network import NEIGHBORS, prefix, stages

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(227142)
    dataset = json.loads((ARCHIVE / 'train.json').read_bytes())['cards']
    terrains = sorted({c['terrain'] for c in dataset})
    assert len(terrains) == 24
    calibration = sorted(terrains, key=lambda h: hashlib.sha256(('block-four/1/cal/' + h).encode()).digest())[:4]
    source = ROOT / 'sources/master3s2/master3s2.py'
    assert sha(source) == SOURCE_SHA
    protocol = dict(schema='block-four-spatial-distillation/1', original=SOURCE_SHA, v2_diagnostic_target='opponent rollout only',
                    files={p.name: sha(p) for p in (
                        ARCHIVE / 'train_boards.zip', ARCHIVE / 'val_boards.zip', ARCHIVE / 'train.json',
                        ARCHIVE / 'val.json', ARCHIVE / 'learn.py', ARCHIVE / 'network.py', Path(__file__))},
                    train_maps=sorted(set(terrains) - set(calibration)), calibration_maps=calibration,
                    network='frozen original blocks 1–4; residual policy head 576×48×8, five axial sites plus global mean',
                    checkpoints=[4, 8, 16, 24, 32], selection='Minimum calibration mean worker and base KL, equal per map',
                    held_target=dict(team_accuracy=.8, worker_accuracy=.94, base_accuracy=.97, worker_KL=.04, base_KL=.04))
    raw = (json.dumps(protocol, sort_keys=True, indent=2) + '\n').encode()
    frozen = HERE / 'protocol.json'
    if args.phase == 'register':
        assert not frozen.exists() or frozen.read_bytes() == raw
        frozen.write_bytes(raw)
        print(sha(frozen), flush=True)
        return
    assert frozen.read_bytes() == raw

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(576, 48), nn.SiLU(), nn.Linear(48, 8))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
            self.register_buffer('mean', torch.zeros(576))
            self.register_buffer('scale', torch.ones(576))

        def forward(self, x, base):
            return base + self.head((x - self.mean) / self.scale)

    @torch.inference_mode()
    def sites(net, x, style, crosscheck=False):
        shallow = prefix(net, x, style)
        deep = stages(net, shallow, net.blocks[2:4])[net.board].permute(1, 0, 2).contiguous()
        slots = x[:, [2, 4, 12]].max(axis=1).reshape(len(x), 225) > 0
        row, cell = np.nonzero(slots)
        i, j = torch.as_tensor(row), torch.as_tensor(cell)
        center = deep[i, j]
        neighbors = NEIGHBORS[j][:, [1, 3, 5, 7]]
        differences = deep[i[:, None], neighbors] - center[:, None]
        features = torch.cat((center, differences.flatten(1), deep.mean(1)[i]), 1)
        base = torch.cat((center @ net.hw[0] + net.hw[1], center @ net.hb[0] + net.hb[1]), 1)
        old = shallow[net.board].permute(1, 0, 2)[i, j] if crosscheck else None
        return row, cell, features, base, old

    def kl(predicted, target, temperature=1):
        return F.kl_div(F.log_softmax(predicted / temperature, 1),
                        F.softmax(target / temperature, 1), reduction='none').sum(1)

    def score(pred, indices, roles, y):
        vals = []
        for lo, hi, role in ((0, 5, 0), (5, 8, 1)):
            idx = indices[roles[indices] == role]
            vals.append(float(kl(pred[idx, lo:hi], y[idx, :hi-lo]).mean()))
        return float(np.mean(vals))

    learn.sites = sites
    m = learn.module(source.read_bytes())
    assert m._net() is m._RNET and len(m._NET.blocks) == 6
    split = 'train' if args.phase == 'fit' else 'val'
    weights_file = HERE / 'weights.npz'
    receipt_file = HERE / 'fit.json'
    if args.phase == 'held':
        receipt = json.loads(receipt_file.read_bytes())
        assert receipt['weights_sha256'] == sha(weights_file)
    xyb, rows, hashes, cards, seconds = learn.extract(m, split, ARCHIVE / (split + '_boards.zip'))
    x, base, y, role, _ = xyb
    head = Student()

    if args.phase == 'fit':
        map_id = np.empty(len(x), dtype=np.int32)
        terrain_ids = {h: i for i, h in enumerate(terrains)}
        for row in rows:
            map_id[row['begin']:row['end']] = terrain_ids[row['terrain']]
        groups = [torch.from_numpy(np.flatnonzero(map_id == terrain_ids[h])) for h in calibration]
        train = torch.from_numpy(np.flatnonzero(~np.isin(map_id, [terrain_ids[h] for h in calibration])))
        head.mean.copy_(x[train].mean(0))
        head.scale.copy_(x[train].std(0).clamp_min(.02))
        opt = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
        checkpoints = []
        best = float('inf')
        for epoch in range(1, 33):
            head.train()
            for ix in train[torch.randperm(len(train))].split(384):
                pred = head(x[ix], base[ix])
                loss = torch.zeros(())
                for lo, hi, kind in ((0, 5, 0), (5, 8, 1)):
                    mask = role[ix] == kind
                    if not mask.any():
                        continue
                    logging_pred, target = pred[mask, lo:hi], y[ix][mask, :hi-lo]
                    loss = loss + kl(logging_pred, target).mean() / 2 + 2 * kl(logging_pred, target, 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch not in protocol['checkpoints']:
                continue
            head.eval()
            with torch.inference_mode():
                pred = head(x, base)
                losses = [score(pred, group, role, y) for group in groups]
            mean_loss = float(np.mean(losses))
            checkpoints.append(dict(epoch=epoch, equal_map_calibration_KL=mean_loss, by_map=losses))
            if mean_loss < best:
                chosen, best = epoch, mean_loss
                saved = {k: v.detach().numpy().copy() for k, v in head.state_dict().items()}
            print('calibration', epoch, mean_loss, flush=True)
        np.savez_compressed(weights_file, **saved)
        learn.write(receipt_file, dict(protocol_sha256=sha(frozen), weights_sha256=sha(weights_file), chosen_epoch=chosen,
                                       checkpoints=checkpoints, boards=len(rows), units=len(x),
                                       training_units=len(train), decode_s=seconds, input_sha256=sorted(hashes)))
    else:
        assert not hashes.intersection(receipt['input_sha256'])
        assert set(c['terrain'] for c in cards).isdisjoint(terrains)
        with np.load(weights_file, allow_pickle=False) as packed:
            head.load_state_dict({k: torch.from_numpy(packed[k]) for k in packed.files})
        head.eval()
        with torch.inference_mode():
            pred = head(x, base)
        metrics, details = learn.report(m, rows, y, pred, 'block4_spatial')
        learn.write(HERE / 'held_orders.json.gz', details)
        previous = json.loads((ARCHIVE / 'val.json').read_bytes())
        assert sorted(hashes) == previous['sample_keys']
        learn.write(HERE / 'held.json', dict(protocol_sha256=sha(frozen), weight_sha256=sha(weights_file),
                                            physical_boards=len(rows), units=len(x), metrics=metrics,
                                            previous_3_block=previous['methods']['block3_spatial']))
        print(metrics['sum'], metrics['map_equal_KL'], flush=True)


if __name__ == '__main__':
    main()

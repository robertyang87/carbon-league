"""Plan stage 3: distill the small RollNet from a trained main checkpoint.

Teacher: a trained MainNet (source-layout npz).  Student: model.RollNet
(2-3 blocks, 32-48 channels, policy heads only).  Data: replay shard
feature planes + unit masks from vec_selfplay.py (raw states as stored;
no symmetry augmentation -- the teacher is not V4-equivariant, and the
student simply mimics the teacher on the states it will actually see).

Objective: temperature-KL with T = 1.5 on both policy heads, masked to
the cells that hold a unit of the side to move:
  L = T^2 * [ KL(softmax(lw_t/T) || softmax(lw_s/T)) over worker cells
            + KL(softmax(lb_t/T) || softmax(lb_s/T)) over base cells ].

Export: a _ROLL npz in the source policy key layout (snames/meta/stem/
emb/blocks/hw/hb), policy keys ONLY -- no value keys, matching what the
chassis' _build(_Policy(_ROLL), None) consumes.  Whether a chassis can
embed it is pack_submission.py's concern (the master3s2 chassis wires
_ROLL; the default fix3 chassis does not).
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--teacher', required=True, help='trained main checkpoint npz')
    ap.add_argument('--data', required=True, help='comma-separated replay dirs')
    ap.add_argument('--out', required=True, help='output _ROLL npz')
    ap.add_argument('--roll-ch', type=int, default=40)
    ap.add_argument('--roll-blocks', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=1.5)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-rows', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import model as model_mod

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    teacher, snames = model_mod.load_main_npz(args.teacher)
    teacher.eval().to(args.device)
    sid = snames.index(layout.SID_NAME) if layout.SID_NAME in snames else len(snames)
    student = model_mod.RollNet(in_ch=54, ch=args.roll_ch, nb=args.roll_blocks,
                                n_sid_rows=len(snames) + 1)
    student.to(args.device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    files = []
    for d in args.data.split(','):
        files += sorted(glob.glob(os.path.join(d, '*.npz')))
    if not files:
        raise SystemExit('no replay shards in %s' % args.data)
    xs, scs, wms, bms = [], [], [], []
    total = 0
    for path in files:
        z = np.load(path)
        xs.append(z['x'])
        scs.append(z['sc'])
        wms.append(z['wmask'])
        bms.append(z['bmask'])
        total += z['x'].shape[0]
        if total >= args.max_rows:
            break
    x_all = np.concatenate(xs)[:args.max_rows]
    sc_all = np.concatenate(scs)[:args.max_rows]
    wm_all = np.concatenate(wms)[:args.max_rows]
    bm_all = np.concatenate(bms)[:args.max_rows]
    print('distilling on %d rows (teacher %s)' % (len(x_all), args.teacher))

    T = args.temperature
    for epoch in range(args.epochs):
        order = rng.permutation(len(x_all))
        tot, nb_ = 0.0, 0
        for s0 in range(0, len(order), args.batch):
            idx = order[s0:s0 + args.batch]
            x = torch.from_numpy(x_all[idx].astype(np.float32) / 100.0).to(args.device)
            wm = torch.from_numpy(wm_all[idx].astype(np.float32)).to(args.device)
            bm = torch.from_numpy(bm_all[idx].astype(np.float32)).to(args.device)
            with torch.no_grad():
                lw_t, lb_t, _, _ = teacher(x, sid, None)
            lw_s, lb_s = student(x, sid)
            logp_w = F.log_softmax(lw_s.float() / T, dim=-1)
            p_w = F.softmax(lw_t.float() / T, dim=-1)
            logp_b = F.log_softmax(lb_s.float() / T, dim=-1)
            p_b = F.softmax(lb_t.float() / T, dim=-1)
            kl_w = -(p_w * logp_w).sum(-1) + (p_w * torch.log(p_w.clamp(min=1e-12))).sum(-1)
            kl_b = -(p_b * logp_b).sum(-1) + (p_b * torch.log(p_b.clamp(min=1e-12))).sum(-1)
            loss = T * T * (((kl_w * wm).sum() / wm.sum().clamp(min=1)) +
                            ((kl_b * bm).sum() / bm.sum().clamp(min=1)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb_ += 1
        print('epoch %d: KL loss %.4f' % (epoch, tot / max(1, nb_)))

    out = student.export_source_dict(snames)
    assert not any(k.startswith(('v1.', 'v2.', 'v3.')) for k in out), 'value keys leaked'
    with open(args.out, 'wb') as f:
        np.savez_compressed(f, **layout.to_f16(out))
    print('wrote %s (policy keys only, %d blocks, %d ch)'
          % (args.out, args.roll_blocks, args.roll_ch))
    return 0


if __name__ == '__main__':
    sys.exit(main())

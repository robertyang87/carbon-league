"""Stage 0 of the KataGo warm-start plan: extract master3s2's weights.

Imports a master3s2 module from a path argument, unpacks its _WEIGHTS /
_VALUE base64 npz payloads, prints the full key/shape/dtype inventory, saves
a numpy mirror (gen0_m3s2.npz, source key layout) and -- when PyTorch is
available -- a torch state_dict (gen0_m3s2.pt) for model.py's MainNet, i.e.
the generation-0 warm-start weights.

Canonical source: src/master3s2.py.  The owner's fix2 and fix3 chassis
(src/apexcarbon_v1_fix2.py, src/apexcarbon_v1_fix3.py) carry
_WEIGHTS/_VALUE payloads that are array-identical to master3s2's
(war-room verified with numpy: same keys, nb=6, same snames, every
array equal), so generation 0 is exactly the net already inside fix3;
master3s2.py stays canonical here because neither chassis has a
_TorchNet class -- the PyTorch forward reference lives only in master3s2.py.

--verify compares, on both random tensors and real encode() outputs:
  * numpy repack: a fresh _Policy built from the re-encoded payload vs the
    module's own _POLICY (runs without torch; the float16 -> float32 ->
    float16 round trip is lossless, so the value arrays are compared for
    exact equality and the policy forward must match to float32 noise);
  * torch MainNet forward vs the source's _Policy.forward (numpy) and vs
    the source's _TorchNet, reporting max abs error (target < 1e-4).

PyTorch is NOT installed on the authoring VM: the numpy parts above always
run; the torch parts fail with a clear message (exit code 3) when torch is
missing, after the inventory and the .npz mirror have been written.
"""

import argparse
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout


def raw_payloads(path):
    """Payloads with their STORED dtypes (float16), for an honest inventory."""
    text = open(path).read()
    pol = layout.decode_npz(layout.load_payload_b64(text, '_WEIGHTS'))
    vb = layout.load_payload_b64(text, '_VALUE')
    val = layout.decode_npz(vb) if vb is not None else None
    return pol, val


def synthetic_players(mod):
    """A small fabricated mid-game position in the env's players format
    (parse(): [cash, {bid: pos}, {wid: [pos, cargo, kind]}, {tid: [pos, age]}]),
    used to exercise encode() on a 'real' input."""
    st = layout.new_game_state(mod, 1234)
    pl = st.players
    pl[0][2] = {'player-0-worker-0': [79, 3, 'COLLECTOR'],
                'player-0-worker-1': [65, 0, 'PLANTER']}
    pl[1][2] = {'player-1-worker-0': [145, 1, 'COLLECTOR'],
                'player-1-worker-1': [155, 0, 'PLANTER']}
    pl[0][3] = {'player-0-tree-0': [94, 5]}
    pl[1][3] = {'player-1-tree-0': [130, 9]}
    pl[0][0] = 137.0
    pl[1][0] = 96.0
    return pl, st.carbon


def verify_numpy_repack(mod, policy32, value32, snames, nb):
    """Rebuild _Policy from re-encoded payloads; compare forwards."""
    pol_b64 = layout.encode_npz(layout.to_f16(
        dict(policy32, snames=np.array(snames), meta=np.array([nb], dtype=np.int32))))
    fresh = mod._Policy(pol_b64)
    rng = np.random.RandomState(42)
    errs = {}
    # random tensors (_Policy.forward is a single-position forward)
    x = (rng.rand(4, mod.NPL, 15, 15) < 0.3).astype(np.float32)
    for i in range(4):
        a = mod._POLICY.forward(x[i], mod._SID)
        b = fresh.forward(x[i], mod._SID)
        errs['random lw'] = max(errs.get('random lw', 0.0), float(np.abs(a[0] - b[0]).max()))
        errs['random lb'] = max(errs.get('random lb', 0.0), float(np.abs(a[1] - b[1]).max()))
    # real encode() outputs (feature-quantized exactly as _act does)
    pl, carbon = synthetic_players(mod)
    for me in (0, 1):
        xe = mod._quant(mod.encode(pl, 40, me, carbon, [1.0] * mod.N)).astype(np.float32)
        a = mod._POLICY.forward(xe, mod._SID)
        b = fresh.forward(xe, mod._SID)
        errs['encode(me=%d) lw' % me] = float(np.abs(a[0] - b[0]).max())
        errs['encode(me=%d) lb' % me] = float(np.abs(a[1] - b[1]).max())
    # value payload round trip is exact (float16 storage)
    if value32 is not None:
        val_b64 = layout.encode_npz(layout.to_f16(value32))
        back = layout.decode_npz(val_b64)
        errs['value roundtrip max'] = max(
            float(np.abs(back[k].astype(np.float32) - value32[k]).max()) for k in back)
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', default=layout.MASTER3S2,
                    help='master3s2 module file (default: %(default)s)')
    ap.add_argument('--out', default=os.path.join(os.getcwd(), 'gen0'),
                    help='output directory (default: ./gen0)')
    ap.add_argument('--verify', action='store_true',
                    help='run the forward-equivalence checks')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pol_raw, val_raw = raw_payloads(args.src)
    snames_raw = pol_raw['snames']
    nb = int(pol_raw['meta'][0])
    print('== payload inventory (stored dtypes) ==')
    inv_pol = {k: v for k, v in pol_raw.items() if k not in ('snames', 'meta')}
    print('\n'.join(layout.inventory_lines(
        inv_pol, val_raw, [str(s) for s in snames_raw], nb)))
    n_par = sum(int(v.size) for v in inv_pol.values())
    print('policy arrays: %d keys, %d elements (float16 as stored)'
          % (len(inv_pol), n_par))

    policy32, value32, snames, nb2 = layout.load_source_payloads(args.src)
    assert nb2 == nb

    # numpy mirror in the source layout: this is the gen0 checkpoint file
    # that train.py / pack_submission.py consume.
    mirror = dict(policy32)
    mirror.update(value32 or {})
    mirror['snames'] = snames_raw
    mirror['meta'] = np.array([nb], dtype=np.int32)
    npz_path = os.path.join(args.out, 'gen0_m3s2.npz')
    with open(npz_path, 'wb') as f:
        np.savez_compressed(f, **layout.to_f16(mirror))
    print('wrote %s' % npz_path)

    mod = layout.load_module(args.src, 'm3s2_extract_src')

    if args.verify:
        errs = verify_numpy_repack(mod, policy32, value32, snames, nb)
        print('== numpy repack verify (fresh _Policy from re-encoded payload) ==')
        for k, v in errs.items():
            print('  max abs err %-18s %.3e' % (k, v))

    # ---- torch part -------------------------------------------------------
    try:
        import torch
    except ImportError:
        print('ERROR: PyTorch is not installed on this machine, so the torch '
              'state_dict (gen0_m3s2.pt) and the _TorchNet comparison cannot '
              'run here.  The numpy inventory, gen0_m3s2.npz and the numpy '
              'repack verify above are complete; run this script on the '
              'training machine (L20) to produce the .pt.', file=sys.stderr)
        return 3

    import model
    net = model.MainNet.from_source(policy32, value32, nb)
    sd = net.state_dict()
    pt_path = os.path.join(args.out, 'gen0_m3s2.pt')
    torch.save(sd, pt_path)
    print('wrote %s (%d tensors)' % (pt_path, len(sd)))

    if args.verify:
        net.eval()
        val_np = {k: value32[k] for k in layout.VALUE_KEYS}
        tnet = mod._TorchNet(mod._POLICY, val_np)
        rng = np.random.RandomState(7)
        cases = {}
        x = (rng.rand(4, mod.NPL, 15, 15) < 0.3).astype(np.float32)
        cases['random'] = (x, rng.randn(4, 5).astype(np.float32))
        pl, carbon = synthetic_players(mod)
        for me in (0, 1):
            xe = mod._quant(mod.encode(pl, 40, me, carbon, [1.0] * mod.N))[None].astype(np.float32)
            sc = mod._scalars(pl, 40, me)[None].astype(np.float32)
            cases['encode(me=%d)' % me] = (xe, sc)
        print('== torch verify: MainNet vs _Policy.forward / _TorchNet (target < 1e-4) ==')
        worst = 0.0
        with torch.no_grad():
            for name, (xx, sc) in cases.items():
                lw_t, lb_t, v_t, _ = net(torch.from_numpy(xx), mod._SID,
                                         torch.from_numpy(sc))
                lw_p = np.stack([mod._POLICY.forward(xx[i], mod._SID)[0]
                                 for i in range(xx.shape[0])])
                lb_p = np.stack([mod._POLICY.forward(xx[i], mod._SID)[1]
                                 for i in range(xx.shape[0])])
                lw_n, lb_n, v_n = tnet(xx, mod._SID, sc)
                e = {
                    'lw vs _Policy': float(np.abs(lw_t.numpy() - lw_p).max()),
                    'lb vs _Policy': float(np.abs(lb_t.numpy() - lb_p).max()),
                    'lw vs _TorchNet': float(np.abs(lw_t.numpy() - lw_n).max()),
                    'lb vs _TorchNet': float(np.abs(lb_t.numpy() - lb_n).max()),
                    'v  vs _TorchNet': float(np.abs(v_t.numpy() - v_n).max()),
                }
                worst = max(worst, max(e.values()))
                for k, val in e.items():
                    print('  %-14s %-16s %.3e' % (name, k, val))
        print('worst max abs err: %.3e  -> %s' % (worst, 'PASS' if worst < 1e-4 else 'FAIL'))
        return 0 if worst < 1e-4 else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

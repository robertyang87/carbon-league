#!/usr/bin/env python3
"""Part 3 quality-gate PILOT: A' vs B' fresh-agent-per-root on rootset32.

Registered design (Part 2 caliber), pilot slice (~42 roots) to prove the
leaf can intervene before any full run:
- Fresh module agent per root per arm, via pack._S.make_search_agent(
  dict(pack.CONF), weights=decoded pack weights, rweights=None,
  oweights=None). This is the identical computation to a fresh pack
  agent() call: pack agent() only adds weight-decode (done once here)
  and _AG caching (defeated by fresh construction per root). Same
  pattern as gatesA/gate2_driver.py and gatesA2/game_driver.py.
- Rootset32 tasks assembled exactly as gate_driver/tasks_from_replays
  (step 200-298, both seats, ACTIVE filter); same per-root seed
  (20261114 + root_index) reset before each arm.
- Serial, unpinned (DGX, no CPUQuota) -- cleanest possible wall-clock
  test of the feeding. No budget/arm/base change of any kind.
- Two samples: A = stratified representative (first game, both seats,
  steps 200,201,202,250,251,252,294,295,296,297,298 = 22 roots);
  B = maximum-power (first 20 Part-2 changed roots, where the fp value
  leaf DID flip the move: 90,192,227,381,404,485,524,530,555,623,769,
  791,989,1088,1187,1286,1287,1484,1486,1558).
- changed := A' action != B' action (byte-compared canonical JSON).
  leaf_called := depth_sum >= min(3, H), H = min(5, 299-step) (sentinel
  rule from the embedded _search source).

Verdict rule (task): if changed == 0 here, the gate is NOT run as a
full changed==0 void gate; report "gate cannot be constructed" with
the depth/leaf-call mechanism evidence.

Usage: qgate_pilot.py <out_json>
Env:   none.
"""
import base64, hashlib, importlib.util, json, os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK_A = os.path.join(HERE, 'echo3_part3_a_pack_v2.py')
PACK_B = os.path.join(HERE, 'echo3_part3_b_pack_v2.py')
ROOTS_DIR = '/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/roots'
ROOTSET = '/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/rootset32.json'

EXP_A_MD5 = 'c6a48797cff10cbd41dc6c3ac5c7dc5c'
EXP_B_MD5 = '3d48742fb489caee2f41a7fc0db3d67c'
EXP_W_MD5 = '444cc54c0d88da5656ff4c7eef1ec8eb'
EXP_MARGIN = 825.4
EXP_A_AFF = '3106.795995527259'
EXP_B_AFF = '567.9139352517781'

SAMPLE_B = [90, 192, 227, 381, 404, 485, 524, 530, 555, 623, 769,
            791, 989, 1088, 1187, 1286, 1287, 1484, 1486, 1558]
SAMPLE_A_STEPS = [200, 201, 202, 250, 251, 252, 294, 295, 296, 297, 298]


def md5file(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def tasks_from_replays(replays_dir, rootset_path):
    import gzip
    games = json.load(open(rootset_path))
    tasks = []
    idx = 0
    for g in games:
        d = json.load(gzip.open(os.path.join(replays_dir, g['key']), 'rt'))
        steps = d['steps']
        for seat in (0, 1):
            for t in range(200, 299):
                if steps[t][seat].get('status') not in (None, 'ACTIVE'):
                    continue
                o0 = steps[t][0]['observation']
                ep = steps[t][seat]['observation']
                obs = {'step': int(o0['step']), 'players': o0['players'],
                       'carbon': ep['carbon'], 'player': seat,
                       'trees': o0.get('trees') or {},
                       'remainingOverageTime': ep.get('remainingOverageTime',
                                                      o0.get('remainingOverageTime'))}
                tasks.append({'game_key': g['key'], 'seat': seat, 'step': t,
                              'root_index': idx, 'obs': obs})
                idx += 1
    return tasks


def load_pack(name, path, exp_md5):
    got = md5file(path)
    assert got == exp_md5, f'{name} pack md5 {got} != {exp_md5}'
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    raw = zlib.decompress(base64.b64decode(m._B1))
    wmd5 = hashlib.md5(raw).hexdigest()
    assert wmd5 == EXP_W_MD5, f'{name} decoded weights md5 {wmd5} != {EXP_W_MD5}'
    w = m._dequantize(raw, m._Q1)
    w['meta'] = np.array(m._M1)
    return m, w


def call(m, w, conf, obs, seed):
    np.random.seed(seed)
    ag = m._S.make_search_agent(conf, weights=w, rweights=None, oweights=None)
    w0 = time.perf_counter(); c0 = time.process_time()
    try:
        a = ag(obs)
        err = None
    except Exception as e:
        a, err = None, repr(e)[:200]
    return a, ag.stats, err, time.perf_counter() - w0, time.process_time() - c0


def main():
    out_json = sys.argv[1]
    mA, wA = load_pack('pack_Ap_qp', PACK_A, EXP_A_MD5)
    mB, wB = load_pack('pack_Bp_qp', PACK_B, EXP_B_MD5)
    confA, confB = dict(mA.CONF), dict(mB.CONF)
    assert confA == confB, f'CONF differs A vs B: {confA} vs {confB}'
    assert confA['SEARCH'] is True and confA['BUDGET'] == 0.7
    assert confA['TEACHER'] == 2 and confA['ROLL'] == 'self'
    assert confA['GUARD'] is True and confA['MODE'] == 'sweep'
    assert mA._S.DEFAULTS['MARGIN'] == EXP_MARGIN, mA._S.DEFAULTS['MARGIN']
    assert mB._S.DEFAULTS['MARGIN'] == EXP_MARGIN, mB._S.DEFAULTS['MARGIN']
    # leaf constants live in the embedded search source text inside the pack
    # (_SRC is a dict {cnet2_feat, cnet2_np, cnet2_search, fastsim})
    def src_text(m):
        s = m._SRC
        if isinstance(s, dict):
            return '\n'.join(v for v in s.values() if isinstance(v, str))
        return s if isinstance(s, str) else ''
    srcA = src_text(mA)
    assert EXP_A_AFF in srcA and EXP_B_AFF in srcA, 'A leaf constants missing'
    srcB = src_text(mB)
    assert EXP_A_AFF not in srcB, 'B unexpectedly carries A leaf constants'
    print('frozen checks PASS: packs, weights 444cc54c, CONF identical, '
          'MARGIN 825.4 both, A leaf constants present, B heuristic', flush=True)

    tasks = tasks_from_replays(ROOTS_DIR, ROOTSET)
    assert len(tasks) == 6336, f'task count {len(tasks)} != 6336'
    by_index = {t['root_index']: t for t in tasks}
    first_game = tasks[0]['game_key']
    sel = []
    for t in tasks:
        if t['game_key'] == first_game and t['step'] in SAMPLE_A_STEPS:
            sel.append((t, 'A'))
    for ri in SAMPLE_B:
        sel.append((by_index[ri], 'B'))
    assert len(sel) == 42, f'selected {len(sel)} != 42'
    assert len({t['root_index'] for t, _ in sel}) == 42, 'duplicate roots'
    print(f'pilot roots: {len(sel)} (A={sum(1 for _, s in sel if s == "A")}, '
          f'B={sum(1 for _, s in sel if s == "B")})', flush=True)

    rows = []
    t_start = time.time()
    for k, (t, sample) in enumerate(sel):
        seed = 20261114 + t['root_index']
        aA, sA, eA, wA_, cA = call(mA, wA, confA, t['obs'], seed)
        aB, sB, eB, wB_, cB = call(mB, wB, confB, t['obs'], seed)
        jA = json.dumps(aA, sort_keys=True) if aA is not None else None
        jB = json.dumps(aB, sort_keys=True) if aB is not None else None
        H = min(5, 300 - t['step'])  # source: left = 299-step, H = min(5, left+1)
        sentinel = min(3, H)
        dA, dB = int(sA.get('depth_sum', 0)), int(sB.get('depth_sum', 0))
        rows.append({'root_index': t['root_index'], 'game_key': t['game_key'],
                     'seat': t['seat'], 'step': t['step'], 'sample': sample,
                     'A_action': jA, 'B_action': jB,
                     'changed': (jA is not None and jB is not None and jA != jB),
                     'A_searched': int(sA.get('searched', 0)),
                     'B_searched': int(sB.get('searched', 0)),
                     'A_depth': dA, 'B_depth': dB,
                     'A_leaf_called': dA >= sentinel,
                     'B_leaf_called': dB >= sentinel,
                     'H': H, 'sentinel': sentinel,
                     'A_shallow': int(sA.get('shallow', 0)),
                     'B_shallow': int(sB.get('shallow', 0)),
                     'A_changed_stat': int(sA.get('changed', 0)),
                     'B_changed_stat': int(sB.get('changed', 0)),
                     'A_errors': int(sA.get('errors', 0)),
                     'B_errors': int(sB.get('errors', 0)),
                     'A_exc': eA, 'B_exc': eB,
                     'A_wall': round(wA_, 4), 'B_wall': round(wB_, 4),
                     'A_cpu': round(cA, 4), 'B_cpu': round(cB, 4)})
        print(f'  {k + 1}/42 root {t["root_index"]} step {t["step"]} '
              f'seat {t["seat"]} [{sample}] changed={rows[-1]["changed"]} '
              f'depth A/B={dA}/{dB} searched A/B={rows[-1]["A_searched"]}/{rows[-1]["B_searched"]}',
              flush=True)
    import collections
    ch = [r for r in rows if r['changed']]
    summ = {'n': len(rows), 'changed': len(ch),
            'changed_A_sample': sum(1 for r in ch if r['sample'] == 'A'),
            'changed_B_sample': sum(1 for r in ch if r['sample'] == 'B'),
            'changed_root_indices': [r['root_index'] for r in ch],
            'A_depth_dist': dict(collections.Counter(r['A_depth'] for r in rows)),
            'B_depth_dist': dict(collections.Counter(r['B_depth'] for r in rows)),
            'A_searched_total': sum(r['A_searched'] for r in rows),
            'B_searched_total': sum(r['B_searched'] for r in rows),
            'A_leaf_called': sum(1 for r in rows if r['A_leaf_called']),
            'B_leaf_called': sum(1 for r in rows if r['B_leaf_called']),
            'A_errors_total': sum(r['A_errors'] for r in rows),
            'B_errors_total': sum(r['B_errors'] for r in rows),
            'A_wall_mean': round(sum(r['A_wall'] for r in rows) / len(rows), 4),
            'B_wall_mean': round(sum(r['B_wall'] for r in rows) / len(rows), 4),
            'elapsed_s': round(time.time() - t_start, 1)}
    out = {'meta': {'pack_A_md5': EXP_A_MD5, 'pack_B_md5': EXP_B_MD5,
                    'weights_md5': EXP_W_MD5, 'margin_prime': EXP_MARGIN,
                    'a_prime': 3106.795995527259, 'b_prime': 567.9139352517781,
                    'conf': confA, 'rootset_md5': 'c0871c50d0cc1eea7e07f8aed6587838',
                    'seeds': '20261114 + root_index (diagnostic domain)'},
           'summary': summ, 'rows': rows}
    json.dump(out, open(out_json, 'w'))
    print('PILOT SUMMARY: ' + json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()

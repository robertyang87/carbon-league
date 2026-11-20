#!/usr/bin/env python3
"""Local stub tests for search_driver pure logic (no torch, no pack)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search_driver as sd


class FakeMod:
    WACT = ['NONE', 'UP', 'RIGHT', 'DOWN', 'LEFT']
    BACT = ['NONE', 'RECCOLLECTOR', 'RECPLANTER']

    @staticmethod
    def parse(players):
        workers = {'w0': (10, 0.0, 'C', 0), 'w1': (20, 0.0, 'P', 0),
                   'w9': (30, 0.0, 'C', 1)}
        bases = [[('b0', 70), ('b1', 154)], [('c0', 3)]]
        return workers, {}, bases, [0.0, 0.0]


def test_onehot():
    tw, tb, wm, bm = sd.onehot_targets(
        FakeMod, None, 0, {'w0': 'RIGHT', 'b0': 'RECCOLLECTOR'})
    assert wm.sum() == 2 and bm.sum() == 2
    assert tw[10, 2] == 1 and tw[10].sum() == 1          # RIGHT
    assert tw[20, 0] == 1                                # absent -> NONE
    assert tw[30].sum() == 0                             # opponent worker excluded
    assert tb[70, 1] == 1                                # RECCOLLECTOR
    assert tb[154, 0] == 1                               # absent -> NONE
    print('onehot OK')


def test_finish_rows():
    rows = [{'margin': None} for _ in range(299)]
    traj = [float(i) for i in range(300)]
    sd.finish_rows(rows, traj)
    assert rows[0]['margin'] == 299.0 and rows[0]['win'] == 1.0
    assert list(rows[0]['aux_delta']) == [15.0, 40.0, 80.0]
    assert list(rows[250]['aux_delta']) == [15.0, 40.0, 49.0]  # clamped at end
    rows2 = [{'margin': None} for _ in range(299)]
    sd.finish_rows(rows2, [-float(i) for i in range(300)])
    assert rows2[0]['win'] == 0.0
    print('finish_rows OK')


def test_decide():
    d = sd.decide(None, {'a': 'UP'}, {'a': 'UP'}, 0.05)
    assert d['searched'] is False and d['agree'] is True and d['issued'] is False
    out = {'best': 3, 'actual_idx': 2, 'val': {(3, 0): 0.9, (2, 0): 0.7},
           'valu': {(3, 0): 0.30, (2, 0): 0.10}, 'depth': 9, 'rolls': 2,
           'alive': [0, 2, 3], 'n_cands': 6}
    d = sd.decide(out, {'a': 'DOWN'}, {'a': 'UP'}, 0.05)
    assert d['searched'] and not d['agree'] and abs(d['adv_u'] - 0.20) < 1e-12
    assert d['issued'] is True
    d = sd.decide(out, {'a': 'DOWN'}, {'a': 'UP'}, 0.25)
    assert d['issued'] is False                            # below threshold
    out2 = dict(out)
    out2['valu'] = {(3, 0): 0.30}                          # actual unvalued
    d = sd.decide(out2, {'a': 'DOWN'}, {'a': 'UP'}, 0.05)
    assert d['adv_u'] is None and d['issued'] is False
    print('decide OK')


def test_segment():
    with tempfile.TemporaryDirectory() as td:
        def shard(path, margins):
            n = len(margins)
            arrs = {k: np.zeros((n,) + {'x': (54, 15, 15), 'sc': (5,),
                                        'tw': (225, 5), 'tb': (225, 3),
                                        'wmask': (225,), 'bmask': (225,),
                                        'carbon': (225,), 'margin': (),
                                        'aux_delta': (3,), 'win': (),
                                        'searched': ()}[k],
                                 dtype=sd.STORE_DTYPES[k]) for k in sd.ROW_KEYS}
            arrs['margin'] = np.array(margins, dtype=np.float32)
            np.savez_compressed(path, **arrs)
        shard(os.path.join(td, 'shard_w0_000.npz'), [5.0] * 299 + [-7.0] * 100)
        shard(os.path.join(td, 'shard_w0_001.npz'), [-7.0] * 199 + [5.0] * 10)
        arrs, segs = sd.segment_stream(td)
        assert [(b - a, m) for a, b, m in segs] == [(299, 5.0), (299, -7.0), (10, 5.0)], segs
    print('segment OK')


if __name__ == '__main__':
    test_onehot()
    test_finish_rows()
    test_decide()
    test_segment()
    print('ALL STUB TESTS PASS')


def test_classify():
    import classify_issue as ci
    recs = []
    for i in range(299):
        recs.append({'step': i, 'searched': True, 'agree': i % 3 == 0,
                     'adv_u': 0.2 if i % 3 == 1 else 0.0})
    c = ci.classify_game(recs, 0.10, 10, 2.0)
    assert c['n_over'] == 100 and abs(c['s_over'] - 20.0) < 1e-9
    assert c['class'] == 'MISTAKE' and c['max_adv'] == 0.2
    recs2 = [{'step': i, 'searched': True, 'agree': True, 'adv_u': 0.0}
             for i in range(299)]
    recs2[7] = {'step': 7, 'searched': True, 'agree': False, 'adv_u': 0.35}
    c2 = ci.classify_game(recs2, 0.10, 10, 2.0)
    assert c2['n_over'] == 1 and c2['class'] == 'SUPPRESSED'  # single spike
    recs3 = [{'step': i, 'searched': True, 'agree': False, 'adv_u': 0.11}
             for i in range(30)] + \
            [{'step': 30 + i, 'searched': True, 'agree': True, 'adv_u': 0.0}
             for i in range(269)]
    c3 = ci.classify_game(recs3, 0.10, 10, 2.0)
    assert c3['n_over'] == 30 and c3['class'] == 'MISTAKE'  # breadth+mass
    print('classify OK')

if __name__ == '__main__':
    test_classify()

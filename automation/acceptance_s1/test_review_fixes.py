"""Regression counterexamples for the 2026-11-14 code review. Local fixtures only."""
import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
CODE = Path(__file__).resolve().parents[2] / 'code'
sys.path.insert(0, str(CODE))
import numpy as np
import torch
import cnet2_build as B
import stage1_teacher_gate as G
from fastsim import FastCarbon
from eval_v53 import CFG
import rl2_checkpoint as CP
import awr_weights as AWR
import cnet2_train as CT


class AwrOwnershipTests(unittest.TestCase):
    def test_weighted_generation_refusal_does_not_modify_healthy_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); gen = root/'generations/gen-A'; gen.mkdir(parents=True)
            fields = {}
            for name, (dtype, shape) in dict(B.FIELDS, w=(np.float16, ())).items():
                arr = np.ones((2,)+shape, dtype=dtype) if name == 'w' else np.zeros((2,)+shape, dtype=dtype)
                np.save(gen/(name+'.npy'), arr)
                fp = gen/(name+'.npy')
                fields[name] = dict(file=fp.name, rows=2, sha256=hashlib.sha256(fp.read_bytes()).hexdigest())
            meta = dict(total=2, games=[dict(key='g0.json.gz', val=False), dict(key='g1.json.gz', val=True)])
            (gen/'meta.json').write_text(json.dumps(meta)); (root/'meta.json').write_text(json.dumps(meta))
            (gen/'manifest.json').write_text(json.dumps(dict(generation='gen-A', total=2, weighted=True, fields=fields)))
            (root/'current.json').write_text(json.dumps(dict(generation='gen-A', dir='generations/gen-A')))
            with contextlib.redirect_stdout(io.StringIO()):
                CT.load(str(root))
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(sys, 'argv', ['awr_weights', str(root), 'unused-corpus']), \
                 patch.object(AWR, 'rescore', side_effect=AssertionError('must reject before deriving')):
                with self.assertRaisesRegex(SystemExit, 'owned by generation'):
                    AWR.main()
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, meta = CT.load(str(root))
            self.assertEqual(meta['derived_layer_check']['mode'], 'generation-owned')


class CheckpointTests(unittest.TestCase):
    def test_full_tree_checks_backbone_optimizer_and_scalars(self):
        for value in ({'stem': torch.tensor([float('nan')])},
                      {'state': {0: {'exp_avg': torch.tensor([float('inf')])}}},
                      {'return': np.array([float('nan')])}):
            with self.assertRaisesRegex(FloatingPointError, 'non-finite'):
                CP.require_finite(value, 'state')
        CP.require_finite({'finite': torch.ones(3), 'absent': None}, 'state')

    def test_interrupted_serialization_preserves_last_recovery_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'last.pt'
            CP.atomic_torch_save({'it': 7, 'net': torch.ones(2)}, path)
            old = path.read_bytes()
            with patch('torch.serialization._save', side_effect=OSError('injected disk failure')):
                with self.assertRaises(OSError):
                    CP.atomic_torch_save({'it': 8}, path)
            self.assertEqual(old, path.read_bytes())
            self.assertEqual(torch.load(path, weights_only=False)['it'], 7)
            self.assertEqual(list(Path(tmp).glob('.state-*')), [])

    def test_checkpoint_marker_hashes_and_previous_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)/'ckpt_last.npz'
            np.savez(actor, w=np.ones(2))
            CP.commit_checkpoint(tmp, {'it': 7}, actor, 'warmup')
            CP.commit_checkpoint(tmp, {'it': 8}, actor, 'train')
            marker = json.loads((Path(tmp)/'checkpoint_complete.json').read_text())
            self.assertEqual(marker['it'], 8)
            for name, digest in marker['files'].items():
                self.assertEqual(digest, hashlib.sha256((Path(tmp)/name).read_bytes()).hexdigest())
            self.assertEqual(torch.load(Path(tmp)/'state_previous.pt', weights_only=False)['it'], 7)


def replay(path, steps=8, collision=False):
    sim = FastCarbon(CFG).reset(seed=41)
    def row(actions):
        out = []
        for p in (0, 1):
            o = sim.obs(p); o['full_carbon'] = list(sim.carbon)
            out.append(dict(observation=o, action=actions[p], status='ACTIVE'))
        return out
    rows = [row([{}, {}])]
    for t in range(40 if collision else steps):
        actions = []
        for p in (0, 1):
            o = sim.obs(p)
            a = {wid: 'UP' for wid in o['players'][p][2]} if not collision else {}
            if t == 0:
                a.update({bid: 'RECCOLLECTOR' for bid in o['players'][p][1]})
            actions.append(a)
        if collision and t:
            w0 = [w for w in sim.workers.values() if w.player == 0]
            w1 = [w for w in sim.workers.values() if w.player == 1]
            if not w0 or not w1:
                break
            a, b = w0[0], w1[0]
            direction = 'UP' if a.pos // 15 != b.pos // 15 else 'LEFT'
            actions[0][a.id] = direction
        sim.step(*actions)
        rows.append(row(actions))
        if collision and sum(len(r['observation']['players'][p][2]) for p, r in enumerate(rows[-1])) < 2 and t:
            break
    with gzip.open(path, 'wt') as fh:
        json.dump({'steps': rows}, fh)
    return len(rows)-1


class TeacherTests(unittest.TestCase):
    def test_q_and_anchor_receive_only_own_prefix_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'g.json.gz'; step = replay(path)
            spec = dict(path=str(path), step=step, me=0, game='g')
            scene = G.extract_scene(spec)
            cfg = dict(K=1, M=1, H=1, seed_teacher=7, seed_cont=9, belief='sample')
            def evaluate(*args, **kwargs):
                self.assertEqual(kwargs['guard_snaps'], (scene['guard_snaps'][0], None))
                self.assertEqual(kwargs['feats'][0].step, step)
                return dict(q=[1.], Q_mean=[.5], Q_std=[0.], arms=[np.zeros(12, dtype=np.int64)],
                            entropy=0., agree_greedy=True, wall_ms=1., guard={},
                            teacher_version=f'v{G.T.TEACHER_VERSION}', belief='sample')
            with patch.object(G, 'extract_scene', return_value=scene), patch.dict(G._W, pol=object()), \
                 patch.object(G.T, 'evaluate_root', side_effect=evaluate) as ev, \
                 patch.object(G, 'continue_game', return_value=.5):
                G._q_scene((spec, cfg)); G._anchor_task((spec, cfg))
                self.assertEqual(ev.call_count, 2)

    def test_actual_prefix_continuation_preserves_first_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'g.json.gz'; step = replay(path)
            scene = G.extract_scene(dict(path=str(path), step=step, me=0, game='g'))
            self.assertTrue(all(len(h) == 8 for h in scene['fm'].hist.values()))
            class Captured(Exception):
                pass
            def capture(_pol, _guard, feat, x, _teacher, _stats):
                self.assertEqual(feat.hist, scene['fm'].hist)
                for key in ('maps', 'hist_cell'):
                    np.testing.assert_array_equal(x[key], scene['x_me'][key])
                self.assertEqual(_guard.snapshot(), scene['guard_snaps'][0])
                raise Captured()
            with patch.object(G, '_guarded_greedy', capture):
                with self.assertRaises(Captured):
                    G.continue_game(None, scene, None, 1)
            self.assertTrue(all(len(h) == 8 for h in scene['fm'].hist.values()))
            state = copy.deepcopy(scene['state'])
            state['wid_counter'] = [50, 60]
            state['tid_counter'] = [70, 80]
            fresh = G._fresh_sim(dict(state=state))
            self.assertEqual(fresh.get_state()['wid_counter'], [50, 60])
            self.assertEqual(fresh.get_state()['tid_counter'], [70, 80])

    def test_public_worker_loss_prefix_and_private_guard_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'collision.json.gz'; step = replay(path, collision=True)
            scene = G.extract_scene(dict(path=str(path), step=step, me=0, game='collision'))
            with gzip.open(path, 'rt') as fh:
                rows = json.load(fh)['steps']
            losses = []
            for p in (0, 1):
                prev = set(); lost = 0
                for row in rows:
                    current = set(row[p]['observation']['players'][p][2])
                    lost += len(prev-current); prev = current
                losses.append(lost)
                self.assertEqual(scene['guard_snaps'][p]['lostw'], lost)
                self.assertEqual(scene['guard_snaps'][p]['prevw'], prev)
            self.assertGreater(sum(losses), 0)
            own, opponent = G._teacher_guard_snaps(scene, 'sample')
            self.assertEqual(own, scene['guard_snaps'][0])
            self.assertIsNone(opponent)


class DirectPool:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, tasks, **kw): return map(fn, tasks)


class BuilderTests(unittest.TestCase):
    def test_actual_merge_quantization_and_corrupt_margins(self):
        for margin, delta in ((7000.12, 0), (-7000.12, 0), (65536.12, 0), (7000.12, 1), (7000.12, float('nan'))):
            with self.subTest(margin=margin, delta=delta), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); src = root/'g.json.gz'
                with gzip.open(src, 'wt') as fh: json.dump({'steps': []}, fh)
                g = dict(file=str(src), key=src.name, names=[B.F.TEACHERS[0]]*2,
                         scores=[max(margin, 0), max(-margin, 0)], sides=[0], val=False, src='fixture',
                         terminal=True, end_reason='horizon', raw_margin=[margin, -margin],
                         raw_u=[B._u_of_margin(margin), B._u_of_margin(-margin)], outcome='normal', steps=299)
                def build(task):
                    data = {k: np.zeros((1,)+shape, dtype=dt) for k, (dt, shape) in B.FIELDS.items()}
                    data.update(margin_raw=np.array([margin+delta], np.float32),
                                u_raw=np.array([B._u_of_margin(margin)], np.float32),
                                outcome_type=np.array([0], np.int8), label_status=np.array([0], np.int8))
                    np.savez(task[5], **data); return task[2], 1, None
                with patch.object(B, '_scan', return_value=([g], [], {})), \
                     patch.object(B, '_game_fingerprint', return_value='fixture'), \
                     patch.object(B, 'ProcessPoolExecutor', DirectPool), patch.object(B, 'build_game', build), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if delta == 0:
                        B.main(str(root/'out'), [])
                    else:
                        with self.assertRaisesRegex(ValueError, 'raw label alignment'):
                            B.main(str(root/'out'), [])


class ReplicaFreshnessTests(unittest.TestCase):
    def test_actual_upstream_attestation_rejects_dead_or_frozen_arms(self):
        source = (CODE/'replica_chain.sh').read_text()
        fragment = source[source.index('ALIVE_R1=0;'):]
        setup = r'''mark() { echo "$*"; }
kill() { [ "$2" != "$DEAD" ]; }
newest_wt() { echo "$1.npz"; }
mt_of() {
 case "$1" in
   *r1.log) echo "$RLOG";;
   *g1.log) echo "$GLOG";;
   r1.npz) echo "$RWEIGHT";;
   g1.npz) echo "$GWEIGHT";;
 esac
}
'''
        for dead, rlog, glog, rw, gw in (('',101,101,101,101), ('g',101,101,101,101),
                                        ('',100,101,101,101), ('',101,101,101,100)):
            env = dict(os.environ, PID_R1='r', PID_G1='g', DEAD=dead,
                       EP0R='1', EP0G='1', PREFIX='fixture', LOGD='/tmp', EPOCHS='2',
                       SNAP_LOG_R1='100', SNAP_LOG_G1='100', SNAP_WT_R1='100', SNAP_WT_G1='100',
                       RLOG=str(rlog), GLOG=str(glog), RWEIGHT=str(rw), GWEIGHT=str(gw))
            cp = subprocess.run(['bash','-u','-c',setup+fragment], env=env, capture_output=True, text=True)
            expected = 0 if not dead and min(rlog,glog,rw,gw)>100 else 1
            with self.subTest(dead=dead, rlog=rlog, gweight=gw):
                self.assertEqual(cp.returncode, expected, cp.stdout+cp.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)

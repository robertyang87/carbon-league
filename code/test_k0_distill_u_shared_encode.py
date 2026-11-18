"""CPU regression for encoding each distillation root once, before candidates.

Run: BC_DEVICE=cpu python code/test_k0_distill_u_shared_encode.py

Exercises the production forward pass and candidate-prefix reconstruction. The reference preserves
the former Br*K encoder path. Tiny deterministic models and real Feat/Prefix
features cover different active slots, candidate prefixes, padding, IPW and an
anchor with different hidden dimensions; they do not measure playing strength.
"""
import copy
import json
import os
from types import SimpleNamespace
import unittest

os.environ["BC_DEVICE"] = "cpu"
os.environ["BC_GPU_AUG"] = "0"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
import torch.nn.functional as Fn

import cnet2_feat as F
from cnet2_model import CNet2
import cnet2_np as NP
import cnet2_train as T
from eval_v53 import CFG
from fastsim import FastCarbon
import k0_distill_u as D
import k0_distill_u_sparse as S

torch.set_num_threads(1)

ENCODER_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
                  "tok_cargo", "hist_cell", "hist_cargo", "hist_move")
STUDENT_CFG = dict(C=8, NB=1, D=16, NH=4, L=1, E=4, G=2, TA=1, TNH=2)
ANCHOR_CFG = dict(C=12, NB=1, D=24, NH=4, L=1, E=8, G=3)


class RecordingNet(CNet2):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.encode_batches = []
        self.head_batches = []
        self.decoded = []

    def encode(self, b):
        self.encode_batches.append(len(b["maps"]))
        return super().encode(b)

    def decode(self, b, tok, ctx):
        logits = super().decode(b, tok, ctx)
        self.decoded.append(logits.detach().clone())
        return logits

    def heads(self, ctx):
        self.head_batches.append(len(ctx))
        return super().heads(ctx)


def model(cfg, seed):
    torch.manual_seed(seed)
    net = RecordingNet(**cfg).train()
    with torch.no_grad():
        # Exercise spatial attention, whose residual output starts at zero.
        for layer in net.tattn:
            layer.out.weight.normal_(0, .03)
            layer.f2.weight.normal_(0, .03)
    for name, p in net.named_parameters():
        if name.startswith(D.FROZEN_PREFIXES):
            p.requires_grad_(False)
    return net


def candidate(x, aux, variant):
    """A legal complete action and its real teacher-forced Prefix features."""
    pf = F.Prefix(x, aux)
    y = np.full(F.OWN, -1, dtype=np.int64)
    uf = np.zeros((F.OWN, F.UF), np.float32)
    mf = np.zeros((F.OWN, 5, F.MF), np.float32)
    legal = np.zeros((F.OWN, 5), bool)
    bases = ((0, 0), (1, 0), (0, 2), (2, 1))[variant]
    for slot in range(F.OWN):
        if not x["tok_mask"][slot]:
            continue
        uf[slot], mf[slot], legal[slot] = pf.feats(slot)
        action = variant % 5 if slot < 10 else bases[slot - 10]
        assert legal[slot, action], (slot, action, variant)
        y[slot] = action
        pf.advance(slot, action)
    return y, uf, mf, legal


def fixture(K=4):
    rows, roots = [], []
    counts = np.array([4, 2, 3], np.int64)
    if K == 1:
        counts[:] = 1
    assert K == 1 or K >= 4
    arms = np.full((3, K, F.OWN), -1, np.int64)
    targets = np.zeros((3, K), np.float64)
    for r, (seat, nw, cash) in enumerate(((0, 0, 80), (1, 2, 50), (0, 4, 90))):
        # Explicit observation fixtures: distinct maps, seats and worker masks.
        obs = FastCarbon(CFG).reset(seed=37 + r, cash=cash).obs(seat)
        obs["players"][seat][2] = {
            f"player-{seat}-worker-{i}": [65 + i, float(7 * i),
                                          "PLANTER" if i % 2 else "COLLECTOR"]
            for i in range(nw)
        }
        obs["players"][1 - seat][2] = {
            f"player-{1-seat}-worker-0": [100, 3.0, "COLLECTOR"]}
        feat = F.Feat(seat)
        obs["step"] = 18 + r
        feat.observe(obs)
        obs["step"] += 1
        x = feat.observe(obs)
        aux = copy.deepcopy(feat.aux)
        logged, uf, mf, legal = candidate(x, aux, 0)
        row = {key: np.array(x[key], copy=True) for key in ENCODER_FIELDS}
        row.update(uf=uf, mf=mf, legal=legal, y=logged,
                   carbon=np.zeros(F.N, np.uint8), teacher=np.int64(1 + 2 * r),
                   gid=np.int64(900 + r), step=np.int64(obs["step"]))
        for k in range(int(counts[r])):
            arms[r, k] = candidate(x, aux, k)[0]
        # Nonuniform target mass, never renormalized by the candidate decoder.
        weights = np.arange(1, counts[r] + 1, dtype=np.float64) ** (1 + r)
        targets[r, :counts[r]] = weights / weights.sum()
        rows.append(row)
        roots.append((copy.deepcopy(x), aux))
    feats = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    return dict(feats=feats, arms=arms, n_arms=counts, target=targets, roots=roots,
                terminal_u=np.array([.5 + .5 / (1 + np.exp(-1.5)),
                                     .5, .5 / (1 + np.exp(.7))], np.float32),
                terminal_wdl=np.array([0, 1, 2], np.int64),
                terminal_margin=np.array([1500, 0, -700], np.float32))


def production_forward(ds, net, anchor):
    K = ds["arms"].shape[1]
    args = SimpleNamespace(kl_beta=.37, lambda_u=.7, lambda_w=.4, lambda_m=.15)
    weights = np.array([1., 4., 2.5])

    def expanded(indices):
        batch, rep, _ = D.expand_candidate_batch(ds["feats"], ds["arms"], indices, K)
        return T.to_dev(batch, aug=False), rep

    def forward(indices):
        batch, _ = expanded(indices)
        return D.dense_forward(net, anchor, batch, ds, ds["target"], weights, indices, args)

    return forward, dict(net=net, anchor=anchor, ds=ds, K=K, args=args,
                         ipw_w=weights, expanded_batch=expanded)


def old_expanded_forward(root_idx, ns):
    """Frozen reference: encode Br*K rows and read value heads from slot zero."""
    b, _ = ns["expanded_batch"](root_idx)
    net, anchor, ds, K = ns["net"], ns["anchor"], ns["ds"], ns["K"]
    tok, ctx, _ = net.encode(b)
    logits = net.decode(b, tok, ctx)
    with torch.no_grad():
        tok_p, ctx_p, _ = anchor.encode(b)
        reference = anchor.decode(b, tok_p, ctx_p)
    idx = np.asarray(root_idx)
    Br = len(idx)
    arms = torch.from_numpy(ds["arms"][idx].reshape(-1, F.OWN))
    joint, kl = D.joint_logp_kl(logits, reference, arms)
    joint, kl = joint.reshape(Br, K), kl.reshape(Br, K)
    ce = -(torch.from_numpy(ds["target"][idx]) * joint).sum(1)
    kl = kl.sum(1) / torch.from_numpy(ds["n_arms"][idx]).double()
    u, wdl, margin = net.heads(ctx)
    mse_u = (u.reshape(Br, K)[:, 0] - torch.from_numpy(ds["terminal_u"][idx])) ** 2
    ce_w = Fn.cross_entropy(wdl.reshape(Br, K, 3)[:, 0],
                           torch.from_numpy(ds["terminal_wdl"][idx]), reduction="none")
    mse_m = (margin.reshape(Br, K)[:, 0]
             - torch.from_numpy(ds["terminal_margin"][idx] / 1000.0)) ** 2
    a = ns["args"]
    loss_root = ce + a.kl_beta * kl + a.lambda_u * mse_u + a.lambda_w * ce_w + a.lambda_m * mse_m
    ipw = torch.from_numpy(ns["ipw_w"][idx])
    loss = (ipw * loss_root).sum() / ipw.sum()
    return loss, ce, kl, joint, mse_u, ce_w, mse_m


class SharedRootEncodingTest(unittest.TestCase):
    def assert_outputs_close(self, got, want):
        self.assertEqual(len(got), len(want))
        for i, (a, b) in enumerate(zip(got, want)):
            self.assertTrue(bool(torch.isfinite(a).all()), f"nonfinite output {i}")
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6, msg=f"output {i}")

    def assert_gradients_close(self, actual, expected):
        gradients = 0
        for (name, p), (other_name, q) in zip(actual.named_parameters(), expected.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-6, msg=name)
                gradients += 1
        self.assertGreater(gradients, 40)
        self.assertGreater(float(actual.stem.weight.grad.abs().max()), 1e-6)
        for head in (actual.utility, actual.wdl, actual.margin, actual.gru):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                for p in head.parameters()))

    def test_outputs_losses_gradients_and_encoder_batch_size(self):
        cases = ((4, STUDENT_CFG, [2, 0, 1]),
                 (4, ANCHOR_CFG, [1, 2, 1, 0]),
                 (1, ANCHOR_CFG, [2, 0, 1]))
        for K, anchor_cfg, indices in cases:
            with self.subTest(K=K, anchor_D=anchor_cfg["D"], indices=indices):
                ds = fixture(K)
                net, anchor = model(STUDENT_CFG, 81), model(anchor_cfg, 92)
                old_net, old_anchor = copy.deepcopy(net), copy.deepcopy(anchor)
                forward, ns = production_forward(ds, net, anchor)
                _, old_ns = production_forward(ds, old_net, old_anchor)
                actual, expected = forward(indices), old_expanded_forward(indices, old_ns)
                self.assert_outputs_close(actual, expected)
                torch.testing.assert_close(net.decoded[0], old_net.decoded[0], rtol=2e-5, atol=2e-6)
                self.assertEqual(net.encode_batches, [len(indices)])
                self.assertEqual(anchor.encode_batches, [len(indices)])
                self.assertEqual(net.head_batches, [len(indices)])
                self.assertEqual(old_net.encode_batches, [len(indices) * K])
                actual[0].backward()
                expected[0].backward()
                self.assert_gradients_close(net, old_net)
                self.assertTrue(all(p.grad is None for p in anchor.parameters()))
                self.assertTrue(all(p.grad is None for p in old_anchor.parameters()))
                for row, idx in enumerate(indices):
                    self.assertTrue(bool((actual[3][row, ds["n_arms"][idx]:] == 0).all()))

    def test_sparse_value_only_roots_do_not_run_the_policy(self):
        ds = fixture(4)
        ds.update(format=S.ANNOTATION_FORMAT_V3, has_q=np.array([1, 0, 1]),
                  info={"max_candidates": 4})
        ds["arms"][1] = -1
        ds["n_arms"][1] = 0
        ds["target"][1] = 0
        net, anchor = model(STUDENT_CFG, 81), model(ANCHOR_CFG, 92)
        def run(indices):
            return S.forward_terms_impl(
                net, anchor, T, ds, ds["target"], np.array([1., 0., 2.]), indices,
                kl_beta=.37, lambda_u=.7, lambda_w=.4, lambda_m=.15,
                dev=torch.device("cpu"), feat_stats={})
        mixed = run([0, 1, 2])
        self.assertEqual(anchor.encode_batches, [2])
        self.assertEqual(net.encode_batches, [2, 1])
        self.assertEqual(mixed["n_q"], 2)
        torch.testing.assert_close(mixed["loss_policy"], run([0, 2])["loss_policy"])
        references_before = len(anchor.encode_batches)
        value_only = run([1])
        self.assertEqual(len(anchor.encode_batches), references_before)
        self.assertTrue(value_only["zero_q"])
        torch.testing.assert_close(value_only["loss"], value_only["loss_value"])
        value_only["loss"].backward()
        self.assertIsNone(net.gru.weight_ih.grad)
        self.assertGreater(float(net.utility.weight.grad.abs().sum()), 0.)

    def test_extra_candidate_padding_preserves_loss_and_gradients(self):
        short, padded = fixture(4), fixture(7)
        net, anchor = model(STUDENT_CFG, 81), model(ANCHOR_CFG, 92)
        other, other_anchor = copy.deepcopy(net), copy.deepcopy(anchor)
        forward, _ = production_forward(short, net, anchor)
        padded_forward, _ = production_forward(padded, other, other_anchor)
        a, b = forward([2, 0, 1]), padded_forward([2, 0, 1])
        self.assert_outputs_close(a[:3] + a[4:], b[:3] + b[4:])
        torch.testing.assert_close(a[3], b[3][:, :4], rtol=2e-5, atol=2e-6)
        self.assertTrue(bool((b[3][:, 4:] == 0).all()))
        a[0].backward()
        b[0].backward()
        self.assert_gradients_close(net, other)

    def test_candidate_features_retain_different_prefixes(self):
        ds = fixture()
        b, rep, _ = D.expand_candidate_batch(ds["feats"], ds["arms"], [2, 0, 1], 4)
        np.testing.assert_array_equal(rep, [2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(set(ds["feats"]["tok_mask"][:, :F.OWN].sum(1)), {2, 4, 6})
        for key in ENCODER_FIELDS:
            for start in (0, 4, 8):
                for offset in range(1, 4):
                    np.testing.assert_array_equal(b[key][start], b[key][start + offset])
        # Root 2 has moving/planting workers; root 1 cannot afford two recruits.
        self.assertFalse(np.array_equal(b["mf"][0], b["mf"][1]))
        self.assertFalse(np.array_equal(b["uf"][0], b["uf"][1]))
        self.assertTrue(b["legal"][8, 11, 1])
        self.assertFalse(b["legal"][9, 11, 1])

    def test_numpy_serving_matches_forced_candidate_logits_and_utility(self):
        ds = fixture()
        net, anchor = model(STUDENT_CFG, 81), model(ANCHOR_CFG, 92)
        weights = {name: p.detach().numpy().copy() for name, p in net.state_dict().items()}
        weights["meta"] = np.array(json.dumps({"cfg": net.cfg}))
        policy = NP.Policy(weights)
        forward, ns = production_forward(ds, net, anchor)
        with torch.no_grad():
            forward([0, 1, 2])
            b, _ = ns["expanded_batch"]([0, 1, 2])
            _, ctx, _ = net.encode({key: val[::4] for key, val in b.items()})
            utilities = net.heads(ctx)[0].numpy()
        logits = net.decoded[0].numpy()
        for r, (x, aux) in enumerate(ds["roots"]):
            for k in range(int(ds["n_arms"][r])):
                arm = ds["arms"][r, k]
                labels, lg, _, utility = policy.act(x, aux, teacher=int(ds["feats"]["teacher"][r]),
                                                     forced=arm, want_u=True)
                np.testing.assert_array_equal(labels, arm)
                active = arm >= 0
                np.testing.assert_allclose(lg[active], logits[r * 4 + k][active], rtol=3e-5, atol=2e-5)
                self.assertAlmostEqual(utility, float(utilities[r]), delta=2e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

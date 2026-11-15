#!/usr/bin/env python3
"""Shared plumbing for the PPO-U value-side offline diagnostic
(PREREG_VALUE, signed text: six checkpoints {74,78,79,82,85,95}).

Everything here mirrors the Stage 1 pipeline (kl_measure.py /
build_roots.py): same root set (roots.npz, 6,336 roots), same batch
assembly, same repo model path (rl2_ppo_cnet2.Cnet2Adapter.make_model).

HEAD_MAP summary (evidence in HEAD_MAP.md): the run trained with
sep_value=1 throughout -- the critic is a SEPARATE CNet2 instance
("vnet") whose state lives only in the trainer state .pt files, NOT
in ckpt_<it>.npz (those export the actor "net" only; the actor's own
value/utility heads are frozen init artifacts in every checkpoint).
Readout era per checkpoint:
  74, 78 : vnet tanh value head  -> vhat = (tanh(v) + 1) / 2
  79, 82, 85, 95 : vnet utility head -> vhat = sigmoid(utility) = u
           (the adapter's utility branch maps u to the reward scale
           as v = 2u - 1, so u IS the [0,1] U prediction)
The end-of-it82 trainer state was not preserved anywhere (rolling
state overwritten; no freeze/backup at 82), so ckpt_82 has no vhat.
"""
import hashlib
import json
import os
import sys

import numpy as np

WORK = "/root/ppo_attrib"
VD = os.path.join(WORK, "value-diag")
STATE_DIR = os.path.join(VD, "state")
OUT = os.path.join(VD, "out")
CODE = "/root/carbon/code"
RUN_DIR = "/root/carbon/data/rl3/snipe_n1"
sys.path.insert(0, CODE)

CKPTS = [74, 78, 79, 82, 85, 95]

# md5 of the Stage 1 ckpt copies (Stage 1 RECEIPT table A) -- verified
# again at execution time per PREREG_VALUE.
CKPT_MD5 = {
    74: "8291f61b492ef63468873d47c6afb3e0",
    78: "198779b558cee796cc681ac6554d66a7",
    79: "f59888d461fc22fae1bcdde043edc826",
    82: "75c4bddb05adc698ad8d694f8bb19081",
    85: "eb633ca99aca432e2f4c9265c368fda8",
    95: "9ece5a33cdc6bc6730c5136d48976a07",
}
# trainer state files (critic carriers): run-dir original -> local copy.
# md5s recorded at prep time (three of them also appear verbatim in the
# swap/freeze watcher logs: it74 683d8758, it78 87cde228, it85 fd9df203).
STATE_SRC = {
    74: "state_pre_termU_it74.pt",   # backup at the Terminal-U switch
    78: "state_pre_vhead_it78.pt",   # backup at the vhead swap
    79: "state_pre_r18_it79.pt",     # backup at the r18 code swap
    85: "state_frozen_it85.pt",      # copy-only freeze at it85
    95: "state_previous.pt",         # atomic-commit previous state (it=95)
}
STATE_MD5 = {
    74: "683d87588ae597920fe20a8c095c6d14",
    78: "87cde228d00bfda6a90d27084e979d8d",
    79: "0b643177c922313eb0895f429232c88b",
    85: "fd9df2039fe95392db49a0836933e32e",
    95: "414b3f20d6a62ad8554bee90314db21a",
}
ROOTS_MD5 = "2f182b96ad97dbedbc591f65a1054d2d"        # /root/ppo_attrib/roots.npz
ROOTSET_MD5 = "c0871c50d0cc1eea7e07f8aed6587838"      # /root/ppo_attrib/rootset32.json

# HEAD_MAP readout per checkpoint: (readout, head weight key, head bias key)
READOUT = {
    74: ("tanh", "value.weight", "value.bias"),
    78: ("tanh", "value.weight", "value.bias"),
    79: ("utility", "utility.weight", "utility.bias"),
    82: ("utility", "utility.weight", "utility.bias"),
    85: ("utility", "utility.weight", "utility.bias"),
    95: ("utility", "utility.weight", "utility.bias"),
}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_arr(a):
    return hashlib.md5(np.asarray(a).tobytes()).hexdigest()


def verify_md5(path, expect, label):
    got = md5_file(path)
    if got != expect:
        raise SystemExit(f"MD5 MISMATCH {label}: {path} got {got} expect {expect}")
    return got


def load_roots():
    z = np.load(os.path.join(WORK, "roots.npz"), allow_pickle=True)
    return z


def batch(z, idx):
    """Stage 1 kl_measure.batch, verbatim rules."""
    import torch
    import cnet2_feat as F
    d = {}
    for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
              "tok_cargo", "hist_cell", "hist_cargo", "hist_move",
              "uf", "mf", "legal", "y", "teacher"):
        a = z[k][idx]
        if a.dtype == np.float16:
            a = a.astype(np.float32)
        d[k] = torch.from_numpy(np.ascontiguousarray(a))
    B = len(idx)
    d["maps"] = (d["maps"].float() / 255.0).reshape(B, F.P, F.S, F.S)
    for k in ("tok_pos", "tok_kind", "hist_cell", "hist_move", "y",
              "teacher"):
        d[k] = d[k].long()
    d["tok_mask"] = d["tok_mask"].bool()
    d["legal"] = d["legal"].bool()
    return d


def build_critic(k, device):
    """Build checkpoint k's critic (the trainer state's vnet) through
    the repo's own adapter path. Architecture cfg comes from the
    paired ckpt_<k>.npz meta (the state's own 'net' arrays are first
    verified to be exactly that npz, so the vnet pairing is anchored
    to the same iteration). Returns (model, info dict)."""
    import torch
    from rl2_ppo_cnet2 import Cnet2Adapter
    spath = os.path.join(STATE_DIR, f"state_it{k}.pt")
    stt = torch.load(spath, map_location="cpu", weights_only=False)
    if stt.get("it") != k:
        raise SystemExit(f"state {spath}: embedded it={stt.get('it')} != {k}")
    cz = np.load(os.path.join(WORK, "state", f"ckpt_{k}.npz"))
    net_sd = stt["net"]
    diffs = 0
    for key in net_sd:
        a = net_sd[key].numpy()
        b = cz[key]
        if a.shape != b.shape or not np.array_equal(a, b):
            diffs += 1
    if diffs:
        raise SystemExit(f"state {spath}: net != ckpt_{k}.npz ({diffs} arrays differ)")
    vnet_sd = stt["vnet"]
    readout, wk, bk = READOUT[k]
    for key in (wk, bk):
        if key not in vnet_sd:
            raise SystemExit(f"state {spath}: vnet lacks {key}")
    weights = {kk: np.asarray(vv.numpy(), np.float32)
               for kk, vv in vnet_sd.items()}
    weights["meta"] = cz["meta"]
    model = Cnet2Adapter().make_model(weights)
    model = model.to(device).eval()
    info = {
        "checkpoint": k,
        "state_file": spath,
        "state_md5": md5_file(spath),
        "embedded_it": int(stt["it"]),
        "net_vs_ckpt_diffs": 0,
        "readout": readout,
        "head_tensors": [wk, bk],
        "head_md5": {wk: md5_arr(weights[wk]), bk: md5_arr(weights[bk])},
        "training_contract": stt.get("training"),
    }
    del stt
    return model, info


def forward_vhat(model, z, readout, idx_all, device, mb=512):
    """vhat for the given root rows + the policy logits of the same
    forward (same batch, same trunk -- 'Stage 1 same forward' rule).
    Returns (vhat float64 [n], logits float32 [n,12,5])."""
    import torch
    vhats, logits = [], []
    with torch.no_grad():
        for i in range(0, len(idx_all), mb):
            idx = idx_all[i:i + mb]
            b = {kk: vv.to(device) for kk, vv in batch(z, idx).items()}
            if readout == "utility":
                lg, _v, _aux, u, _wdl, _mk = model.forward_heads(b)
                vh = u
            else:
                lg, v, _aux = model(b)
                vh = (v + 1.0) / 2.0
            vhats.append(vh.double().cpu().numpy())
            logits.append(lg.float().cpu().numpy())
    return np.concatenate(vhats), np.concatenate(logits)

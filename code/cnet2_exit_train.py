"""cnet2_exit_train -- expert-iteration distillation of cnet2_search into the cnet2 net (MPS).

Mixes the BC dataset (cnet2_build format, e.g. data/cnet2_ds_v4) with search data (exit_merge format).
Per batch: (1-MIX) BC samples + MIX search samples.
  BC samples:     hard CE on the teacher action (teacher weights as in cnet2_train x per-sample w).
  search samples: hard CE on the PLAYED action; slots where played != net greedy (changed) x CH_W;
                  at the searched unit's slot the hard target is replaced by the soft target pi
                  (softmax of root values), weight SOFT_W (x CH_W if changed).
  value: MSE on tanh(margin/3000) for both (VAL_W);  aux carbon head as in cnet2_train.
Usage: python cnet2_exit_train.py BC_DIR EXIT_DIR OUT.npz EPOCHS
env: BC_INIT (warm start npz), BC_BS 512, BC_LR 3e-4, EX_MIX 0.5, EX_CH_W 3, EX_SOFT_W 1, EX_VAL_W 0.5,
     BC_MAXSTEPS, EX_STEPS_PER_EPOCH (default: one pass over the search train set)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnet2_feat as F  # noqa: E402
import cnet2_train as CT  # noqa: E402
from cnet2_build import FIELDS  # noqa: E402
from cnet2_model import CNet2  # noqa: E402
from exit_gen import EXTRA  # noqa: E402

DEV = CT.DEV
MIX = float(os.environ.get("EX_MIX", "0.5"))
CH_W = float(os.environ.get("EX_CH_W", "3"))
SOFT_W = float(os.environ.get("EX_SOFT_W", "1"))
VAL_W = float(os.environ.get("EX_VAL_W", "0.5"))
SAMPLED_W = float(os.environ.get("EX_SAMPLED_W", "0.3"))     # opening steps whose net move was temperature-sampled
LOADK = [k for k in FIELDS if k not in ("gid", "step")]


def _sha256_path(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_EXIT_ROW_KEYS = ("n_rows", "rows", "num_rows", "total_rows", "row_count", "n")


def load_exit(dirn):
    keys = list(FIELDS) + list(EXTRA)
    # D1 (Grok review, P1): the exit half of this trainer used to np.load
    # every flat file bare -- no row cross-check, no finite scan, no
    # identity record -- while the BC half of the same main() goes
    # through cnet2_train.load's full gate stack. The checks below are
    # the trainer-side minimum for the flat exit format (the exit data
    # product itself carries no generation/manifest): identity hashes
    # of the exact bytes on disk, a field row-count cross-check, and a
    # finite scan, all before anything trains on the data.
    meta_raw = open(os.path.join(dirn, "meta.json"), "rb").read()
    meta = json.loads(meta_raw)
    field_sha = {k: _sha256_path(os.path.join(dirn, k + ".npy")) for k in keys}
    D = {k: np.load(os.path.join(dirn, k + ".npy"), mmap_mode="r") for k in keys}
    # Row-count cross-check: every field must agree with the row count
    # meta declares (when meta carries one) and with every other field;
    # a torn or mixed flat directory dies here, naming the field.
    declared = next((meta[k] for k in _EXIT_ROW_KEYS
                     if isinstance(meta.get(k), int) and not isinstance(meta.get(k), bool)), None)
    if declared is not None:
        ref, ref_src = int(declared), "meta declared row count"
    else:
        ref, ref_src = int(np.asarray(D["gid"]).shape[0]), "field 'gid'"
    for k in keys:
        n_k = int(np.asarray(D[k]).shape[0])
        if n_k != ref:
            raise SystemExit(
                f"exit data row-count mismatch in {dirn!r}: field {k!r} has {n_k} rows, "
                f"expected {ref} (from {ref_src}); refusing to train")
    # Finite scan: any NaN/Inf in a float field dies here, naming the
    # field (the BC half's F1 gate has refused such data all along).
    for k in keys:
        a = D[k]
        if np.issubdtype(a.dtype, np.floating):
            bad = int((~np.isfinite(np.asarray(a))).sum())
            if bad:
                raise SystemExit(
                    f"exit data non-finite entries in {dirn!r} field {k!r}: "
                    f"{bad} NaN/Inf; refusing to train")
    is_val = np.array([g["val"] for g in meta["games"]])[np.asarray(D["gid"])]
    print(json.dumps({"exit_data_identity": {
        "dir": dirn,
        "meta_sha256": hashlib.sha256(meta_raw).hexdigest(),
        "fields": field_sha,
        "rows": ref,
        "val_rows": int(is_val.sum()),
        "games": len(meta["games"]),
    }}), flush=True)
    return D, is_val, meta


def _canonical_sha256(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verify_exit_generation(gen_dir, ptr):
    # D1 seal (producer side: exit_merge): a sealed generation is consumed
    # verify-or-refuse, BEFORE load_exit's own gates -- the pointer's hash
    # of the manifest bytes, the manifest's self-seal, the pointer/manifest
    # generation match, the meta seal, then every declared field file
    # (existence, byte size, sha256, row count), each refusal naming the
    # item. Nothing is loaded from the generation until all of it passes.
    man_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(
            f"sealed exit generation has no manifest.json: {gen_dir!r}; refusing to train")
    man_raw = open(man_path, "rb").read()
    want = ptr.get("manifest_sha256")
    if want is not None and hashlib.sha256(man_raw).hexdigest() != want:
        raise SystemExit(
            f"exit manifest hash mismatch in {gen_dir!r}: manifest.json bytes do not match "
            f"the pointer's manifest_sha256; refusing to train")
    man = json.loads(man_raw)
    if man.get("generation") != ptr.get("generation"):
        raise SystemExit(
            f"exit pointer/manifest generation mismatch: pointer {ptr.get('generation')!r} "
            f"vs manifest {man.get('generation')!r}; refusing to train")
    core = {k: v for k, v in man.items() if k != "self_sha256"}
    if man.get("self_sha256") != _canonical_sha256(core):
        raise SystemExit(
            f"exit manifest self-seal mismatch in {gen_dir!r}: manifest content does not "
            f"match its self_sha256; refusing to train")
    meta_p = os.path.join(gen_dir, "meta.json")
    if not os.path.exists(meta_p) or _sha256_path(meta_p) != man.get("meta_sha256"):
        raise SystemExit(
            f"exit generation meta seal mismatch in {gen_dir!r}: meta.json does not match "
            f"manifest meta_sha256; refusing to train")
    for k in sorted(man.get("fields", {})):
        info = man["fields"][k]
        p = os.path.join(gen_dir, info["file"])
        if not os.path.exists(p):
            raise SystemExit(
                f"exit generation file missing: field {k!r} ({p!r}); refusing to train")
        if os.path.getsize(p) != info["bytes"]:
            raise SystemExit(
                f"exit generation file size mismatch: field {k!r} has "
                f"{os.path.getsize(p)} bytes, manifest declares {info['bytes']}; "
                f"refusing to train")
        if _sha256_path(p) != info["sha256"]:
            raise SystemExit(
                f"exit generation file hash mismatch: field {k!r} ({info['file']}); "
                f"refusing to train")
        rows = int(np.load(p, mmap_mode="r").shape[0])
        if rows != info["rows"]:
            raise SystemExit(
                f"exit generation file row mismatch: field {k!r} has {rows} rows, "
                f"manifest declares {info['rows']}; refusing to train")
    print(json.dumps({"exit_data_sealed": {
        "generation": man.get("generation"), "dir": gen_dir,
        "format": man.get("format"),
        "files_verified": len(man.get("fields", {})),
    }}), flush=True)
    return gen_dir


def resolve_exit_arg(arg):
    # Consumption paths for the EXIT_DIR argument (D1 spec):
    # - pointer path: `arg` is a current.json pointer file, or a dataset
    #   root containing current.json -> resolve to the sealed generation
    #   and verify it in full (the ONLY way to consume a sealed product);
    # - a sealed generation directory passed directly is refused -- it
    #   must come through its pointer, so the verified generation is the
    #   one the pointer certifies, never a lookalike directory;
    # - direct path: an unsealed legacy flat directory loads as before
    #   under the transition gates, but NAMED (exit_data_legacy_unsealed).
    if os.path.isfile(arg):
        ptr = json.load(open(arg))
        root = os.path.dirname(os.path.abspath(arg))
        return _verify_exit_generation(os.path.join(root, ptr["dir"]), ptr)
    if os.path.exists(os.path.join(arg, "current.json")):
        ptr = json.load(open(os.path.join(arg, "current.json")))
        return _verify_exit_generation(os.path.join(arg, ptr["dir"]), ptr)
    if os.path.exists(os.path.join(arg, "manifest.json")):
        raise SystemExit(
            f"sealed exit generation at {arg!r} must be consumed via its dataset root / "
            f"current pointer, not by direct path; refusing to train")
    print(json.dumps({"exit_data_legacy_unsealed": {
        "dir": arg,
        "note": "no manifest/current pointer: unsealed legacy exit product, "
                "loading under transition gates (named per D1 spec)",
    }}), flush=True)
    return arg


def batch_np(Dbc, ibc, Dex, iex):
    ibc = np.sort(ibc); iex = np.sort(iex)
    b = {}
    for k in LOADK:
        b[k] = np.concatenate([np.asarray(Dbc[k][ibc]), np.asarray(Dex[k][iex])])
    nb, ne = len(ibc), len(iex)
    w = np.asarray(Dbc["w"][ibc], np.float32) if "w" in Dbc else np.ones(nb, np.float32)
    b["w"] = np.concatenate([w, np.ones(ne, np.float32)]).astype(np.float32)
    b["is_ex"] = np.concatenate([np.zeros(nb, bool), np.ones(ne, bool)])
    pi = np.zeros((nb + ne, F.OWN, 5), np.float32); pi[nb:] = np.asarray(Dex["pi"][iex], np.float32)
    ps = np.full(nb + ne, -1, np.int64); ps[nb:] = np.asarray(Dex["pi_slot"][iex])
    yn = b["y"].astype(np.int64).copy(); yn[nb:] = np.asarray(Dex["y_net"][iex])
    smp = np.zeros(nb + ne, bool)
    if "sampled" in Dex:
        smp[nb:] = np.asarray(Dex["sampled"][iex])
    b["pi"] = pi; b["pi_slot"] = ps; b["y_net"] = yn; b["smp"] = smp
    return b


def to_dev(b, aug):
    """cnet2_train.to_dev + the same symmetry applied to pi and y_net."""
    pi, ps, yn, isx, smp = b.pop("pi"), b.pop("pi_slot"), b.pop("y_net"), b.pop("is_ex"), b.pop("smp")
    B = len(b["y"])
    g = torch.randint(0, 4, (B,)) if aug else torch.zeros(B, dtype=torch.long)
    t = CT.to_dev(dict(b), aug=False)
    if aug:
        cm = CT.SYM_CELL[g]; mm = CT.SYM_MOVE[g]
        dv = DEV
        cmd = cm.to(dv); mmd = mm.to(dv)
        t["maps"] = torch.gather(t["maps"].reshape(B, F.P, F.N), 2, cmd[:, None, :].expand(-1, F.P, -1)).reshape(B, F.P, F.S, F.S)
        t["carbon"] = torch.gather(t["carbon"], 1, cmd)
        t["tok_pos"] = torch.gather(cmd, 1, t["tok_pos"])
        hc = t["hist_cell"]
        hcm = torch.gather(cmd, 1, hc.clamp(min=0).reshape(B, -1)).reshape(hc.shape)
        t["hist_cell"] = torch.where(hc >= 0, hcm, hc)
        mm6 = torch.cat([mmd, torch.full((B, 1), 5, dtype=torch.long, device=dv)], 1)
        t["hist_move"] = torch.gather(mm6, 1, t["hist_move"].reshape(B, -1)).reshape(t["hist_move"].shape)
        mf = t["mf"]
        wv = torch.gather(mf[:, :10], 2, mmd[:, None, :, None].expand(-1, 10, -1, F.MF))
        t["mf"] = torch.cat([wv, mf[:, 10:]], 1)
        for key in ("y",):
            y = t[key]
            yw = torch.where(y[:, :10] >= 0, torch.gather(mmd, 1, y[:, :10].clamp(min=0)), y[:, :10])
            t[key] = torch.cat([yw, y[:, 10:]], 1)
        ynd = torch.from_numpy(yn).to(dv)
        ynw = torch.where(ynd[:, :10] >= 0, torch.gather(mmd, 1, ynd[:, :10].clamp(min=0)), ynd[:, :10])
        ynd = torch.cat([ynw, ynd[:, 10:]], 1)
        pid = torch.from_numpy(pi).to(dv)
        pw = torch.gather(pid[:, :10], 2, mmd[:, None, :].expand(-1, 10, -1))
        pid = torch.cat([pw, pid[:, 10:]], 1)
    else:
        ynd = torch.from_numpy(yn).to(DEV); pid = torch.from_numpy(pi).to(DEV)
    t["y_net"] = ynd; t["pi"] = pid; t["pi_slot"] = torch.from_numpy(ps).to(DEV); t["is_ex"] = torch.from_numpy(isx).to(DEV); t["smp"] = torch.from_numpy(smp).to(DEV)
    return t


def losses(net, b):
    if CT.AMP == "bf16" and DEV.type == "mps":
        with torch.autocast("mps", dtype=torch.bfloat16):
            lg, v, aux = net(b)
        lg, v, aux = lg.float(), v.float(), aux.float()
    else:
        lg, v, aux = net(b)
    y = b["y"]; valid = y >= 0
    logp = torch.log_softmax(lg, -1)
    hard = -torch.gather(logp, 2, y.clamp(min=0)[..., None])[..., 0]          # [B,OWN]
    B = y.shape[0]
    isx = b["is_ex"]
    tw = torch.ones(B, device=y.device)
    for k, wgt in CT.TW.items():
        tw = torch.where((b["teacher"] == k) & ~isx, torch.full_like(tw, wgt), tw)
    tw = tw * b["w"]
    slot_w = valid.float() * tw[:, None]
    chg = (y != b["y_net"]) & valid & isx[:, None]
    slot_w = slot_w * torch.where(chg, torch.full_like(slot_w, CH_W), torch.ones_like(slot_w))
    # sampled opening moves are noisy targets: down-weight them (search-changed slots keep their weight)
    slot_w = slot_w * torch.where(b["smp"][:, None] & ~chg, torch.full_like(slot_w, SAMPLED_W), torch.ones_like(slot_w))
    # soft target at the searched slot
    ps = b["pi_slot"]
    has = isx & (ps >= 0)
    onehot = Fn.one_hot(ps.clamp(min=0), F.OWN).bool() & has[:, None]       # [B,OWN]
    soft = -(b["pi"] * torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))).sum(-1)
    per = torch.where(onehot, SOFT_W * soft, hard)
    lpi = (per * slot_w).sum() / slot_w.sum().clamp(min=1)
    lv = Fn.mse_loss(v, b["value"])
    la = Fn.mse_loss(aux, b["carbon"]) * 10.0
    return lpi, lv, la, lg


@torch.no_grad()
def evaluate(net, Dbc, vbc, Dex, vex, bs=1024):
    net.eval()
    out = {}
    for name, Dx, idx, ex in (("bc", Dbc, vbc, False), ("exit", Dex, vex, True)):
        ok = n = okc = nc = 0; vl = 0.0; nv = 0
        for i in range(0, len(idx), bs):
            ii = idx[i:i + bs]
            b = batch_np(Dbc, ii[:0] if ex else ii, Dex, ii if ex else ii[:0])
            t = to_dev(b, aug=False)
            lpi, lv, la, lg = losses(net, t)
            pred = lg.argmax(-1); y = t["y"]; m = (y >= 0) & (t["tok_kind"][:, :F.OWN] < 2)
            ok += ((pred == y) & m).sum().item(); n += m.sum().item()
            c = m & (y != t["y_net"])
            okc += ((pred == y) & c).sum().item(); nc += c.sum().item()
            vl += lv.item() * len(y); nv += len(y)
        out[name] = {"workers_acc": round(ok / max(n, 1), 4), "changed_acc": round(okc / max(nc, 1), 4), "n_changed": nc,
                     "value_mse": round(vl / max(nv, 1), 4)}
    net.train()
    return out


def main(bc_dir, ex_dir, out, epochs):
    torch.manual_seed(0); np.random.seed(0)
    Dbc, vbc_mask, _ = CT.load(bc_dir)
    Dex, vex_mask, meta = load_exit(resolve_exit_arg(ex_dir))
    tbc = np.nonzero(~vbc_mask)[0]; vbc = np.nonzero(vbc_mask)[0]
    tex = np.nonzero(~vex_mask)[0]; vex = np.nonzero(vex_mask)[0]
    rs = np.random.RandomState(1)
    vbc = np.sort(rs.choice(vbc, min(len(vbc), 20000), replace=False))
    init = os.environ.get("BC_INIT")
    w0 = np.load(init) if init else None
    cfg = json.loads(str(w0["meta"]))["cfg"] if w0 is not None else {}
    net = CNet2(**cfg).to(DEV)
    if w0 is not None:
        net.load_state_dict({k: torch.from_numpy(w0[k]) for k in w0.files if k != "meta"})
    bs = int(os.environ.get("BC_BS", "512")); lr = float(os.environ.get("BC_LR", "3e-4"))
    ne = max(1, int(round(bs * MIX))); nbc = bs - ne
    spe = int(os.environ.get("EX_STEPS_PER_EPOCH", "0")) or max(1, len(tex) // ne)
    if int(os.environ.get("BC_MAXSTEPS", "0")):
        spe = min(spe, int(os.environ["BC_MAXSTEPS"]))
    print(json.dumps({"bc_train": len(tbc), "bc_val": len(vbc), "ex_train": len(tex), "ex_val": len(vex), "cfg": net.cfg,
                      "bs": bs, "mix": MIX, "ch_w": CH_W, "soft_w": SOFT_W, "steps_per_epoch": spe}), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=0.05)
    log = open(out.replace(".npz", "_log.jsonl"), "a")
    r0 = evaluate(net, Dbc, vbc, Dex, vex)
    print(json.dumps({"ep": -1, "val": r0}), flush=True); log.write(json.dumps({"ep": -1, "val": r0}) + "\n")
    best = None
    for ep in range(epochs):
        t0 = time.time(); acc = np.zeros(3); n = 0
        pe = np.random.permutation(tex)
        for i in range(spe):
            iex = pe[(i * ne) % len(pe):(i * ne) % len(pe) + ne]
            if len(iex) < ne:
                iex = np.random.choice(tex, ne)
            ibc = np.random.choice(tbc, nbc)
            b = to_dev(batch_np(Dbc, ibc, Dex, iex), aug=True)
            lpi, lv, la, _ = losses(net, b)
            loss = lpi + VAL_W * lv + la
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step(); sched.step()
            if i % 20 == 0:
                acc += [lpi.item(), lv.item(), la.item()]; n += 1
            if i % 200 == 0:
                print(json.dumps({"ep": ep, "i": i, "pi": round(acc[0] / n, 4), "v": round(acc[1] / n, 4), "s": round(time.time() - t0)}), flush=True)
        r = evaluate(net, Dbc, vbc, Dex, vex)
        rec = {"ep": ep, "val": r, "train_pi": round(acc[0] / max(n, 1), 4), "s": round(time.time() - t0)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        CT.export(net, out.replace(".npz", "_ep%d.npz" % ep), {"cfg": net.cfg, "epoch": ep, "val": r})
        CT.export(net, out, {"cfg": net.cfg, "epoch": ep, "val": r, "data": [bc_dir, ex_dir]})


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]))

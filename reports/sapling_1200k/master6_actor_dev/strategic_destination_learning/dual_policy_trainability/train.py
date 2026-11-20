"""One declared optimizer step on archived, legal, post-search teacher replacements."""
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np
import torch
import corpus
import runtime as R

HERE = Path(__file__).resolve().parent
PLAN_SHA = "1de8f866702b4a3f14acdfd3a47da4219ea6112010d07d426d233d74807fd141"
TOWERS = ("own_it50", "sapling")


def atomic(path, content):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def encode(x):
    return (json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def physical(root, y):
    return corpus.physical(root, y)


def supervised_slots(root, parent, table):
    picks = []
    for slot, (legal, _) in table.items():
        target = int(root["actual"][slot])
        if root["changed"][slot] and root["eligible"][slot] and legal[target] and parent[slot] != target:
            picks.append(slot)
        if parent[slot] != target:
            break
    return picks


def summarize_clock(times):
    a = np.asarray(times) * 1000
    return dict(n=len(times), mean_ms=float(a.mean()), p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()))


def timed(clock, fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    clock.append(time.perf_counter()-t0)
    return result


def main():
    if R.sha(HERE / "protocol.json") != PLAN_SHA:
        raise ValueError("protocol changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1049)
    roots = corpus.roots()
    if len(roots) != 64 or {r["gid"] for r in roots} != {0, 2}:
        raise ValueError("unexpected source panel")
    mod, original, nets = R.load()
    R.set_native(mod, original)
    runtime_source = R.inspect.getsource(mod._return_decode)
    old_decoder = mod._return_decode
    old_orders = [mod._mixed_actions(r["x"], r["aux"]) for r in roots]
    native = R.record_native(mod)
    clocks = defaultdict(list)
    parity = dict(legal_logits=0, exact_legal_logits=0, max_abs_error=0., max_centered_error=0.,
                  branch_orders_compared=0, complete_equal=0, best_branch_counts={"baseline":0, "residual":0})
    rows, eligible = [], []
    for root, old_y in zip(roots, old_orders, strict=True):
        y, ref = timed(clocks["native_recorded_before"], native, root)
        with torch.no_grad():
            fitted, tables, chosen = timed(clocks["torch_before"], R.decode_pair, mod, nets, root)
        if not np.array_equal(old_y, y) or not np.array_equal(y, fitted):
            raise ValueError(("initial complete action mismatch", root["gid"], root["step"]))
        parity["complete_equal"] += 1
        parity["best_branch_counts"]["residual" if chosen else "baseline"] += 1
        for branch in (False, True):
            parity["branch_orders_compared"] += 1
            if set(ref[branch]) != set(tables[branch]):
                raise ValueError("different active slots")
            for s, (legal, z) in ref[branch].items():
                valid, predicted = tables[branch][s]
                if not np.array_equal(legal, valid):
                    raise ValueError("different legal actions")
                target, pred = z[legal], predicted.detach().numpy()[legal]
                if target.argmax() != pred.argmax():
                    raise ValueError("initial branch argmax differs")
                err = pred-target
                parity["legal_logits"] += len(err)
                parity["exact_legal_logits"] += int(np.count_nonzero(err == 0))
                parity["max_abs_error"] = max(parity["max_abs_error"], float(abs(err).max()))
                parity["max_centered_error"] = max(parity["max_centered_error"], float(abs(err-err[0]).max()))
        slots = supervised_slots(root, y, ref[chosen])
        eligible.append(slots)
        rows.append(dict(gid=root["gid"], seed=root["seed"], seat=root["seat"], step=root["step"], split=root["split"],
                         parent=physical(root,y), teacher=physical(root,root["actual"]), taught_slots=slots,
                         branch="residual" if chosen else "baseline"))
    if parity["max_abs_error"] > 1e-4 or parity["max_centered_error"] > 1e-4:
        raise ValueError(("initial legal logits fail", parity))
    n = sum(len(s) for r,s in zip(roots,eligible,strict=True) if r["split"] == "fit")
    if not n:
        raise ValueError("no authenticated post-search disagreement under matching parent prefix")
    optimizer = torch.optim.AdamW([p for net in nets for p in net.parameters()], lr=.00005, weight_decay=0)
    training_loss = 0.
    t0 = time.perf_counter()
    for r, slots in zip(roots, eligible, strict=True):
        if r["split"] != "fit" or not slots:
            continue
        y, tables, chosen = R.decode_pair(mod, nets, r)
        loss = 0
        for s in slots:
            valid, z = tables[chosen][s]
            loss = loss - z.masked_fill(~torch.from_numpy(valid),-torch.inf).log_softmax(0)[int(r["actual"][s])] / n
        training_loss += float(loss.detach())
        loss.backward()
    optimizer.step()
    update_seconds = time.perf_counter()-t0
    quantized, fp32, towers = [], [], {}
    patch_buffers = []
    patch_rows = []
    for key, net, old, keys in zip(TOWERS, nets, original, R.KEYS, strict=True):
        new = {k: a.detach().numpy().copy() for k, a in net.state_dict().items()}
        before = R.zlib.decompress(R.base64.b64decode(getattr(mod, keys[0])))
        layout, raw, deq = R.cnet2_pack.quantize(new)
        if layout != getattr(mod,keys[1]):
            raise ValueError("int8 layout drift")
        delta = np.bitwise_xor(np.frombuffer(before,np.uint8),np.frombuffer(raw,np.uint8)).tobytes()
        packed = zlib.compress(delta,9)
        if np.bitwise_xor(np.frombuffer(before,np.uint8),np.frombuffer(zlib.decompress(packed),np.uint8)).tobytes()!=raw:
            raise ValueError("candidate parameter patch did not round-trip")
        patch_rows.append(dict(tower=key,start=sum(map(len,patch_buffers)),length=len(packed),
                               raw_sha256=hashlib.sha256(raw).hexdigest(), original_raw_sha256=hashlib.sha256(before).hexdigest()))
        patch_buffers.append(packed)
        g={k:p.grad.detach() for k,p in net.named_parameters() if p.grad is not None}
        towers[key] = dict(tensors=len(new), parameters=sum(v.size for v in new.values()),
            nonzero_gradient_parameters=sum(int(torch.count_nonzero(a)) for a in g.values()),
            gradient_l2=float(sum(a.double().square().sum() for a in g.values()).sqrt()),
            stem_gradient_l2=float(g["stem.weight"].double().norm()),
            changed_fp32=sum(int(np.count_nonzero(v!=old[k])) for k,v in new.items()),
            changed_exported=sum(int(np.count_nonzero(v!=old[k])) for k,v in deq.items()))
        quantized.append(deq)
        fp32.append(new)
    if not all(v["gradient_l2"]>0 and v["stem_gradient_l2"]>0 and v["changed_exported"]>0 for v in towers.values()):
        raise ValueError("optimizer step failed to reach both native towers")
    costs = dict()
    mod._return_decode = old_decoder
    for name, arrays in (("parent", original), ("updated_fp32", fp32), ("updated_int8", quantized)):
        R.set_native(mod, arrays)
        for r,row in zip(roots,rows,strict=True):
            y=timed(clocks[name+"_numpy_decoder"],mod._mixed_actions,r["x"],r["aux"])
            p=physical(r,y)
            if name=="parent" and row["parent"]!=p:
                raise ValueError("unmodified decoder differs")
            row[name]=p
    # Measure the unchanged objective in FP32 and int8 on fixed original prefixes.
    for name, arrays in (("parent",original),("updated_fp32",fp32),("updated_int8",quantized)):
        R.set_native(mod,arrays)
        totals = defaultdict(lambda:[0.,0])
        for r,slots,old_y,row in zip(roots,eligible,old_orders,rows,strict=True):
            if not slots:
                continue
            loss = pinned_nll(mod,r,old_y,slots,row["branch"]=="residual")
            totals[r["split"]][0]+=sum(loss)
            totals[r["split"]][1]+=len(loss)
        costs[name]={s:{"n":v[1],"mean_NLL":v[0]/v[1]} for s,v in totals.items()}
    changes={}
    for split in ("fit","probe"):
        subset=[r for r in rows if r["split"]==split]
        changes[split]={}
        for first,last in (("parent","updated_fp32"),("parent","updated_int8"),("updated_fp32","updated_int8")):
            pairs=[(r[first],r[last]) for r in subset]
            changes[split][first+"_to_"+last]={
                "roots":len(pairs),"different_full_orders":sum(a!=b for a,b in pairs),
                "changed_physical_unit_orders":sum(sum(a.get(k)!=b.get(k) for k in a.keys()|b.keys()) for a,b in pairs)}
    atomic(HERE/"int8_delta.bin",b"".join(patch_buffers))
    atomic(HERE/"orders.json.gz",gzip.compress(encode(rows),mtime=0))
    answer=dict(protocol_sha256=PLAN_SHA,parent_sha256=R.sha(R.SOURCE),sources=corpus.sources(),
                inputs={x:R.sha(HERE/x) for x in ("train.py","runtime.py","corpus.py")},
                exact_zero_update_quantized_roundtrip=True,initial=parity,train_labels=n,
                initial_teacher_NLL=training_loss,post_step_fixed_prefix_NLL=costs,
                update_wall_seconds=update_seconds, gradient_towers=towers, action_changes=changes,
                diagnostic_warm_timing={k:summarize_clock(v) for k,v in clocks.items()},
                patch=dict(file="int8_delta.bin",sha256=R.sha(HERE/"int8_delta.bin"),towers=patch_rows),
                cpu_affinity=sorted(os.sched_getaffinity(0)),original_decoder_sha256=hashlib.sha256(runtime_source.encode()).hexdigest(),
                orders_sha256=R.sha(HERE/"orders.json.gz"))
    atomic(HERE/"result.json",encode(answer))
    print(json.dumps({k:answer[k] for k in ("initial","train_labels","gradient_towers","action_changes","post_step_fixed_prefix_NLL")}),flush=True)


def pinned_nll(mod, root, parent_y, slots, residual):
    x, aux = root["x"], root["aux"]
    z0 = [p.act(x,aux,teacher=mod._HYBRID["style"],forced=parent_y) for p in mod._NETS]
    feat = R.sys.modules["cnet2_feat"]
    prefix = feat.Prefix(x,aux)
    out=[]
    for s in np.flatnonzero(x["tok_mask"][:feat.OWN]):
        _,movement,legal=prefix.feats(s)
        if s in slots:
            z=.75*z0[0][1][s]+.25*z0[1][1][s]
            if residual and s<10 and x["tok_kind"][s]==0:
                z=z+mod._return_basis(x,aux,s,movement)@mod._RETURN_THETA
            g=z[legal].astype(float);m=g.max()
            out.append(float(m+np.log(np.exp(g-m).sum())-z[int(root["actual"][s])]))
        prefix.advance(s,int(parent_y[s]))
    return out


if __name__ == "__main__":
    main()

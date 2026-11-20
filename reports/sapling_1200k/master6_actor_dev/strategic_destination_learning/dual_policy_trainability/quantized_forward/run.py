"""Matched FP versus quantized-forward training on fixed public prefixes and expert labels."""
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import torch
from torch.func import functional_call

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import corpus
import runtime as R
from train import atomic, encode
from quantization import fake_quantize, parameters_for

PLAN = "4fade31557cf0475bbacf58639e2bae6453cab56f9bb4715dc40375f36004b51"
NAMES = ("own_it50","sapling")


def state(net):
    return {k: v.detach().numpy() for k,v in net.state_dict().items()}


def state_hash(nets):
    checksum = hashlib.sha256()
    for net in nets:
        for name,tensor in sorted(state(net).items()):
            checksum.update(name.encode())
            checksum.update(tensor.tobytes())
    return checksum.hexdigest()


def split_batch(roots, rows, name, mod):
    xs = []
    slots = []
    for i,(r,row) in enumerate(zip(roots,rows,strict=True)):
        if row["split"]!=name or not row["taught_slots"]:
            continue
        if (r["seed"],r["step"],r["seat"]) != (row["seed"],row["step"],row["seat"]):
            raise ValueError("root identity mismatch")
        x = r["x"]
        b = R.tensor_batch(x,mod._HYBRID["style"])
        orig = mod._mixed_actions(x,r["aux"])
        if corpus.physical(r,orig)!=row["parent"]:
            raise ValueError("native parent changed its conditioning prefix")
        prefix = R.sys.modules["cnet2_feat"].Prefix(x,r["aux"])
        uf = np.zeros((12,9),np.float32)
        mf = np.zeros((12,5,15),np.float32)
        legal = np.zeros((12,5),bool)
        residual = np.zeros((12,5),np.float32)
        for s in np.flatnonzero(orig>=0):
            uf[s],mf[s],legal[s] = prefix.feats(s)
            if row["branch"]=="residual" and s<10 and x["tok_kind"][s]==0:
                residual[s] = mod._return_basis(x,r["aux"],s,mf[s]) @ mod._RETURN_THETA
            prefix.advance(s,int(orig[s]))
        b.update(y=torch.tensor(orig)[None], uf=torch.tensor(uf)[None], mf=torch.tensor(mf)[None],
                 legal=torch.tensor(legal)[None], residual=torch.tensor(residual)[None])
        for s in row["taught_slots"]:
            y = int(r["actual"][s])
            if y==orig[s] or not legal[s,y]:
                raise ValueError("invalid supervision")
            slots.append((len(xs),s,y,i))
        xs.append(b)
    return {k:torch.cat([b[k] for b in xs],0) for k in xs[0]}, slots


def forward(nets, batch, use_quant):
    outputs=[]
    for net in nets:
        if use_quant:
            outputs.append(functional_call(net, parameters_for(net), (batch,), strict=True)[0])
        else:
            outputs.append(net(batch)[0])
    return .75*outputs[0]+.25*outputs[1]+batch["residual"]


def lattice_check(net):
    _,_,dequantized = R.cnet2_pack.quantize(state(net))
    for k,q in parameters_for(net).items():
        if not np.array_equal(q.detach().numpy(), dequantized[k]):
            raise ValueError(("training quantization differs from real packer", k))
    return dequantized


def objective(z, batch, slots):
    b,s,y,_ = zip(*slots)
    z = z[b,s].masked_fill(~batch["legal"][b,s],-torch.inf)
    return torch.nn.functional.cross_entropy(z,torch.tensor(y))


def audit(nets, mod, prepared, roots):
    """Compare legal Torch QAT values against real packer -> real loaded NumPy tower on locked prefixes."""
    with torch.no_grad():
        d = [lattice_check(net) for net in nets]
        R.set_native(mod,d)
        results = {}
        for split,(batch,slots) in prepared.items():
            z = forward(nets,batch,True)
            worst = 0.
            native_losses=[]
            native_logit_count = 0
            for b,s,target,i in slots:
                x,aux=roots[i]["x"],roots[i]["aux"]
                forced=batch["y"][b].numpy()
                z0,z1=[p.act(x,aux,teacher=mod._HYBRID["style"],forced=forced)[1][s] for p in mod._NETS]
                lg=.75*z0+.25*z1+batch["residual"][b,s].numpy()
                mask=batch["legal"][b,s].numpy()
                worst=max(worst,float(abs(lg[mask]-z[b,s].numpy()[mask]).max()))
                native_logit_count+=int(mask.sum())
                log_partition=np.logaddexp.reduce(lg[mask].astype(float))
                native_losses.append(float(log_partition-lg[target]))
            if worst>1e-4:
                raise ValueError(("Torch deployed-forward error",worst))
            results[split]=dict(n=len(slots),native_legal_logits=native_logit_count,mean_NLL=float(np.mean(native_losses)),
                                torch_mean_NLL=float(objective(z,batch,slots)),max_abs_service_logit_error=worst)
    return results


def serialize(mod,nets,tag):
    blocks=[]
    cards=[]
    for k, net, ids in zip(NAMES,nets,R.KEYS,strict=True):
        orig=zlib.decompress(base64.b64decode(getattr(mod,ids[0])))
        layout,raw,_=R.cnet2_pack.quantize(state(net))
        if layout!=getattr(mod,ids[1]):
            raise ValueError("export layout differs")
        patch=np.bitwise_xor(np.frombuffer(orig,np.uint8),np.frombuffer(raw,np.uint8))
        compressed=zlib.compress(patch.tobytes(),9)
        cards.append(dict(tower=k,start=sum(map(len,blocks)),length=len(compressed),
                          raw_sha256=hashlib.sha256(raw).hexdigest(),original_raw_sha256=hashlib.sha256(orig).hexdigest()))
        blocks.append(compressed)
    name=tag+"_int8_delta.bin"
    atomic(HERE/name,b"".join(blocks))
    return dict(file=name,sha256=R.sha(HERE/name),towers=cards)


def run():
    if R.sha(HERE/"protocol.json")!=PLAN:
        raise ValueError("QAT protocol changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1049)
    t0=time.perf_counter()
    roots=corpus.roots()
    original_result=json.loads((HERE.parent/"result.json").read_bytes())
    order_file=HERE.parent/"orders.json.gz"
    if R.sha(order_file)!=original_result["orders_sha256"]:
        raise ValueError("prior registered labels changed")
    rows=json.loads(gzip.decompress(order_file.read_bytes()))
    mod,arrays,initial=R.load()
    R.set_native(mod,arrays)
    prepared={k:split_batch(roots,rows,k,mod) for k in ("fit","probe")}
    if [len(prepared[k][1]) for k in ("fit","probe")] != [11,12]:
        raise ValueError("initial supervision rows changed")
    result=dict(protocol_sha256=PLAN,code={n:R.sha(HERE/n) for n in ("run.py","quantization.py")},
                parent_result_sha256=R.sha(HERE.parent/"result.json"),arms={})
    step_one_hash=None
    for arm in ("fp","qat"):
        nets=copy.deepcopy(initial)
        optimizer=torch.optim.AdamW([p for net in nets for p in net.parameters()],lr=.00005,weight_decay=0)
        steps=[]
        b,slots=prepared["fit"]
        for step in range(1,5):
            for net in nets:
                lattice_check(net)
            optimizer.zero_grad(set_to_none=True)
            logits=forward(nets,b,arm=="qat")
            loss=objective(logits,b,slots)
            loss.backward()
            gradients={name:float(sum(p.grad.double().square().sum() for p in n.parameters() if p.grad is not None).sqrt())
                       for name,n in zip(NAMES,nets,strict=True)}
            optimizer.step()
            after=audit(nets,mod,prepared,roots)
            receipt=dict(step=step,optimizer_input_NLL=float(loss.detach()),gradient_l2=gradients,exported=after)
            if step==1:
                checksum=state_hash(nets)
                if arm=="fp":
                    step_one_hash=checksum
                if checksum!=step_one_hash:
                    raise ValueError("first step broke the identity control")
                receipt["identical_step_one_fp32_sha256"]=checksum
            steps.append(receipt)
        for r,row in zip(roots,rows,strict=True):
            y=mod._mixed_actions(r["x"],r["aux"])
            row[arm]=corpus.physical(r,y)
        result["arms"][arm]=dict(steps=steps,nll=steps[-1]["exported"],patch=serialize(mod,nets,arm))
    result["physical"]={}
    for s in ("fit","probe"):
        rs=[r for r in rows if r["split"]==s]
        result["physical"][s]={}
        for a,b in (("parent","fp"),("parent","qat"),("fp","qat")):
            result["physical"][s][a+"_to_"+b]=dict(
                maps=sorted({r["seed"] for r in rs}),roots=len(rs),
                different_orders=sum(r[a]!=r[b] for r in rs),
                unit_differences=sum(sum(r[a].get(k)!=r[b].get(k) for k in r[a].keys()|r[b].keys()) for r in rs))
    atomic(HERE/"commands.json.gz",gzip.compress(encode(rows),mtime=0))
    result["orders_sha256"]=R.sha(HERE/"commands.json.gz")
    result["wall_seconds"]=time.perf_counter()-t0
    atomic(HERE/"result.json",encode(result))
    print(json.dumps({k:result[k] for k in ("physical","wall_seconds")}),flush=True)


if __name__=="__main__":
    run()

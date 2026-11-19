"""Distinguish changes to integer codes, row scales and released floating-point values."""
import base64
import zlib
from common import *

def decomposed(m):
    raw=zlib.decompress(base64.b64decode(m._W64))
    entries={};pos=0
    for k,shape,typ,n in m._QMETA:
        size=int(np.prod(shape))
        if typ=="int8":
            q=np.frombuffer(raw,np.int8,count=size,offset=pos)
            scales=np.frombuffer(raw,"<f4",count=shape[0],offset=pos+size)
        elif typ=="f32":
            q=np.frombuffer(raw,"<f4",count=size,offset=pos);scales=np.empty(0)
        else:
            raise ValueError(("unexpected deployment storage",typ))
        entries[k]=(typ,q,scales)
        pos+=n
    if pos!=len(raw):raise ValueError("storage length")
    return entries

def group(k):
    layer=k.split(".")[0]
    if layer in ("stem","stem_n","blocks","tattn"):return "convolution_and_board_attention"
    if layer in ("tok1","tok2","fuse","attn","sc1","sc2","ctx","teach"):return "tokens_and_context"
    if layer in ("h0","din","aemb","gru"):return "recurrent_decoder"
    if layer in ("whead","bhead","U","V"):return "action_readouts"
    if layer in ("aux","value","margin","wdl","utility"):return "auxiliary_and_value"
    raise ValueError(k)

def audit():
    from collections import Counter
    m=package();baseline=decomposed(m);rec=read(HERE/"fit.json");result={}
    for arm in ("full","head"):
        path=HERE/(arm+".py");n=package(path);encoded=decomposed(n)
        if sha(path)!=rec['arms'][arm]["package_sha256"] or n._QMETA!=m._QMETA:
            raise ValueError("different release")
        by={}
        with np.load(BASE) as w0,np.load(HERE/(arm+".npz")) as w1:
            for key,(typ,q,scales) in baseline.items():
                t,qi,si=encoded[key]
                counts=by.setdefault(group(key), Counter())
                counts[typ+"_codes_or_values_changed"]+=int(np.count_nonzero(q!=qi))
                counts["row_scales_changed"]+=int(np.count_nonzero(scales!=si))
                counts["source_fp32_parameters_changed"]+=int(np.count_nonzero(w0[key]!=w1[key]))
                v=m._NETS[1].w[key].astype(float)
                diff=n._NETS[1].w[key]-v
                counts["released_value_changes"]+=int(np.count_nonzero(diff))
                counts["squared_l2_delta"]+=float(np.sum(diff*diff))
                counts["parent_squared_l2"]+=float(np.sum(v*v))
                counts["total_parameters"]+=v.size
        for counts in by.values():
            counts["relative_l2"]=float(np.sqrt(counts["squared_l2_delta"]/max(counts["parent_squared_l2"],1e-20)))
        if sum(c["released_value_changes"] for c in by.values())!=rec['arms'][arm]['quantized_changed']:
            raise ValueError("release change count disagrees")
        result[arm]=by
    save(HERE/"weight_audit.json",dict(fit_sha=sha(HERE/'fit.json'),audit_sha=sha(Path(__file__)),groups=result,
        definition="INT8-code changes, per-output-row scale changes, FP32 payload changes in nonquantized tensors, original FP32 archive changes and dequantized effective differences are counted separately. A scale change can move many effective values without changing their integer codes."))
if __name__=="__main__":audit()

"""Embed the two old-map economic predictors into the fixed 7cf5 root search."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"ladder/tree_payback"
def sha(b):return hashlib.sha256(b).hexdigest()
def build():
    parent=(HERE.parent/"cold_start/candidate.py").read_bytes();assert sha(parent)=="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
    w=json.loads((HERE/"weights.json").read_bytes());risk=json.loads((OLD/"weights.json").read_bytes())["full"]
    payload="\n_TREE_LIB={}\nexec("+repr((OLD/"runtime.py").read_text())+",_TREE_LIB)\n_TREE_RISK="+repr(risk)+"\n_TREE_CASH="+repr({k:w[k] for k in ("feature_mean","feature_scale","means","coef")})+"\n"+(HERE/"runtime.py").read_text()+"\n"
    marker=b"def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    entry=b"    if len(cands) < 2:\r\n"
    addition=payload.replace("\n","\r\n").encode()
    call=b"    _tree_menu_impl(obs, configuration, act0, cands, most)\r\n"
    assert parent.count(marker)==parent.count(entry)==1
    package=parent.replace(marker,addition+marker).replace(entry,call+entry)
    assert package.replace(addition,b"").replace(call,b"")==parent and len(package)<10_000_000
    return package
if __name__=="__main__":
    b=build();print(len(b),sha(b))

"""Pack the trained opponent forecast with the frozen Sapling mixture and produce an identity manifest."""
import hashlib
import json
from pathlib import Path
from build_it50_endhome import PARENT,PARENT_SHA

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/"all_top5_pressure_dev"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if digest(PARENT)!=PARENT_SHA:raise ValueError("unexpected parent")
    model=json.loads((OUT/"model.json").read_bytes());src=PARENT.read_text()
    patches=[("def _mixed_actions(x, aux):","def _pressure_encode(x):"),
             ("    prefix = F.Prefix(x, aux)","    return states\n\n\ndef _pressure_decode(x, aux, encoded, changes=None):\n    F = sys.modules['cnet2_feat']\n    states = [list(s) for s in encoded]\n    prefix = F.Prefix(x, aux)"),
             ("        labels[slot] = action","        if changes is not None:\n            action = _pressure_challenge(F, x, prefix, slot, action, combined, legal, changes)\n        labels[slot] = action")]
    for old,new in patches:
        if src.count(old)!=1:raise ValueError("unexpected decoder")
        src=src.replace(old,new)
    original=(HERE/"it50_endhome_runtime.py").read_text()
    joint=original[original.index("def _return_destinations"):original.index("def _mixed_actions")]
    data=(src+"\n"+joint+"\n"+(HERE/"pressure_return_features.py").read_text()+"\n_PRESSURE_COEF=np.array("+repr(model["coefficients"])+", dtype=float)\n"+(HERE/"pressure_return_runtime.py").read_text()).encode()
    dest=PARENT.with_name("pressure_return_it50_v3.py")
    compile(data,str(dest),"exec")
    if len(data)>10_000_000 or (dest.exists() and dest.read_bytes()!=data):raise ValueError("invalid or differently frozen package")
    dest.write_bytes(data)
    files=[PARENT,dest,HERE/"it50_endhome_runtime.py",Path(__file__),HERE/"pressure_return_train.py",HERE/"pressure_return_features.py",HERE/"pressure_return_runtime.py",OUT/"plan.json",OUT/"model.json",OUT/"split.json"]
    ids={str(f.relative_to(ROOT)):digest(f) for f in files}
    (OUT/"build_manifest.json").write_text(json.dumps({"files_sha256":ids},indent=2)+"\n")
    print(digest(dest),flush=True)
if __name__=="__main__":main()

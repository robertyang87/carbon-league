"""Build the income-regression variant from the two frozen neural policies."""
import hashlib
import json
from pathlib import Path
from build_it50_endhome import PARENT, PARENT_SHA

HERE=Path(__file__).resolve().parent
OUTPUT=PARENT.with_name("orchard_it50_v3.py")


def build():
    raw=PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA: raise ValueError("unexpected parent")
    src=raw.decode()
    patches=[
        ("def _mixed_actions(x, aux):","def _orchard_encode(x):"),
        ("    prefix = F.Prefix(x, aux)", "    return states\n\n\ndef _orchard_decode(x, aux, encoded, changes=None):\n    F = sys.modules['cnet2_feat']\n    states = [list(state) for state in encoded]\n    prefix = F.Prefix(x, aux)"),
        ("        labels[slot] = action", "        if changes is not None:\n            action = _orchard_decide(F, x, prefix, slot, action, combined, legal, changes)\n        labels[slot] = action")]
    for old,new in patches:
        if src.count(old)!=1: raise ValueError("unexpected decoder")
        src=src.replace(old,new)
    original=(HERE/"it50_endhome_runtime.py").read_text()
    joint=original[original.index("def _return_destinations"):original.index("def _mixed_actions")]
    model=json.loads((HERE/"orchard_learned.json").read_text())["model"]
    src+="\n"+joint+"\n"+(HERE/"orchard_value.py").read_text()
    src+="\n_ORCHARD_MODELS = "+repr(model)+"\n"+(HERE/"orchard_runtime.py").read_text()
    compile(src,str(OUTPUT),"exec")
    data=src.encode()
    if len(data)>10_000_000: raise ValueError("size limit")
    if OUTPUT.exists() and data!=OUTPUT.read_bytes(): raise ValueError("candidate already frozen")
    OUTPUT.write_bytes(data)
    print(hashlib.sha256(data).hexdigest())

if __name__=="__main__":
    build()

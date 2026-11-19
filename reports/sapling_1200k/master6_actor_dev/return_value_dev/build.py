import json
from protocol import HERE,sha
from pathlib import Path
PARENT=HERE.parents[3]/"data/weights/sapling_1200k/terminal_return_fit_v1.py"
def build():
    p=json.loads((HERE/"protocol.json").read_bytes())
    if sha(HERE/"route.py")!=p["model_sha256"] or sha(PARENT)!=p["parent_pack_sha256"]:raise ValueError("policy or parent drift")
    parent=PARENT.read_bytes()
    extra=("_RV_MODEL=types.ModuleType('tax_route')\nexec("+repr((HERE/"route.py").read_text())+",_RV_MODEL.__dict__)\n_RV_W="+repr(json.loads((HERE/"weights.json").read_bytes()))+"\n").encode()+(HERE/"runtime.py").read_bytes()
    path=HERE/"candidate.py"; pack=parent+extra
    if len(pack)>10_000_000:raise ValueError("pack size")
    compile(pack,str(path),"exec")
    if path.exists() and path.read_bytes()!=pack: raise ValueError("immutable candidate mismatch")
    path.write_bytes(pack)
    files=[HERE/x for x in ("protocol.json","route.py","runtime.py","build.py","train.py","weights.json","candidate.py")]
    manifest=dict(parent_sha256=sha(PARENT),size=len(pack),sha256={f.name:sha(f) for f in files})
    (HERE/"release.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__":build()

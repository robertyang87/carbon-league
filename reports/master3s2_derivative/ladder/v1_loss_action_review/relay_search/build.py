import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parents[2]
def sha(v):return hashlib.sha256(v).hexdigest()
if __name__=="__main__":
    parent=(D/"cold_start/candidate.py").read_bytes()
    assert sha(parent)=="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
    piece=b"\n"+(HERE/"runtime.py").read_bytes()
    candidate=parent+piece
    assert candidate[:-len(piece)]==parent and len(candidate)<=10_000_000
    (HERE/"candidate.py").write_bytes(candidate)
    receipt=dict(parent=sha(parent),append=sha(piece),candidate=sha(candidate),bytes=len(candidate),preserves_parent_prefix_bytes=len(parent))
    (HERE/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt)

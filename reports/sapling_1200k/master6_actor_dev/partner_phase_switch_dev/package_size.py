"""Measure alternate encodings without constructing an unfitted adaptive candidate or changing its weights."""
import ast
import base64
import hashlib
from register import HERE,PARTNER,ROOT,sha,seal

def blobs(path):
    text=path.read_text()
    nodes={t.targets[0].id:t for t in ast.parse(text).body if isinstance(t,ast.Assign) and len(t.targets)==1 and isinstance(t.targets[0],ast.Name)}
    return {k:ast.literal_eval(nodes[k].value) for k in ("_B1","_W64","_Q1","_M1")}

def compact(payload):
    return repr(payload).encode("utf-8")

def main():
    p=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py';q=PARTNER/'candidate.py'
    old,new=blobs(p),blobs(q)
    assert old["_W64"]==new["_W64"]
    table=[];savings=0
    for name,enc in [("it50",old["_B1"]),("it20",new["_B1"]),("sapling",old["_W64"])]:
        raw=base64.b64decode(enc); alt=base64.b85encode(raw)
        assert base64.b85decode(alt)==raw
        size=len(compact(enc));alt_size=len(compact(alt.decode("ascii")));savings+=size-alt_size
        table.append({"name":name,"compressed_weights_sha256":hashlib.sha256(raw).hexdigest(),"base64_literal_bytes":size,"base85_literal_bytes":alt_size})
    # Include the old pack in full and add the second expert literal, metadata and 16 KiB selector+decoder space.
    estimate=p.stat().st_size+len(compact(new["_B1"]))+len(compact(new["_Q1"]))+len(compact(new["_M1"]))
    seal(HERE/"package_size.json",{"it50_pack_sha256":sha(p),"it20_pack_sha256":sha(q),"data":table,
        "base64_two_expert_size_before_selector":estimate,"conservative_base85_size_including_16KiB_code_allowance":estimate-savings+16384,
        "scope":"Byte estimate on unchanged compressed payloads, not an installed pack or runtime certification. If empirical stages pass a real selector pack needs actual byte size, numerical parity, startup/resource/time measurement."})
if __name__=="__main__":main()

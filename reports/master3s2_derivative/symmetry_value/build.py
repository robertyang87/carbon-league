"""Derive the symmetry-value package from our frozen cold-start model by two byte insertions."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/"cold_start/candidate.py"
EXPECTED="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    raw=PARENT.read_bytes()
    assert sha(raw)==EXPECTED
    rotate=(HERE/"rotate.py").read_text()
    tree=ast.parse(rotate)
    method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="rotate_features")
    transform=ast.get_source_segment(rotate,method)
    helper=transform+"\n\n"+(HERE/"readout.py").read_text()+"\n\n"
    block=helper.replace("\n","\r\n").encode()
    mark=b"def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    spot=b"        if t == D2 or not n:                # as deep as asked, or the game is over on every board"
    call=b"        if t == D2 and n:\r\n            _symmetry_values(live, pol, alive, val, dep, net, me, t0)\r\n"
    assert raw.count(mark)==1 and raw.count(spot)==1
    candidate=raw.replace(mark,block+mark).replace(spot,call+spot)
    assert candidate.replace(block,b"").replace(call,b"")==raw
    compile(candidate,"symmetry_value.py","exec")
    return candidate


if __name__=="__main__":
    candidate=build()
    (HERE/"candidate.py").write_bytes(candidate)
    print(sha(candidate))

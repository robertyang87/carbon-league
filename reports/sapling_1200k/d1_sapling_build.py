"""Build the frozen hybrid actor into Echo4 D1's H=1 comparator for a separate development trial."""
import argparse
import ast
import hashlib
from pathlib import Path

from build_it50_endhome import PARENT, PARENT_SHA

HERE = Path(__file__).resolve().parent
D1 = HERE.parent / "search_dossier/iter2/echo4_search_int8_d1.py"
OUTPUT = PARENT.with_name("d1_sapling.py")
D1_MD5 = "e8ea1e0e95ca76aa88f330c18df1d1a5"


def literal_definitions(source, keys):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in keys}


def patch(source, before, after):
    if source.count(before) != 1:
        raise ValueError("D1 source anchor must match exactly once: " + before)
    return source.replace(before, after)


def build(output):
    parent, d1 = PARENT.read_bytes(), D1.read_bytes()
    if hashlib.sha256(parent).hexdigest() != PARENT_SHA or hashlib.md5(d1).hexdigest() != D1_MD5:
        raise ValueError("parent or D1 identity mismatch")
    text = d1.decode()
    hybrid = literal_definitions(parent.decode(), {"_B1", "_M1", "_Q1", "_W64", "_QMETA", "_WMETA", "_SRC"})
    original = literal_definitions(text, {"_SRC", "CONF"})
    src, cfg = original["_SRC"], original["CONF"]
    if any(src[key] != hybrid["_SRC"][key] for key in ("cnet2_feat", "cnet2_np")):
        raise ValueError("feature or policy implementation differs")
    cfg["GUARD"] = False
    search = src["cnet2_search"]
    search = patch(search, 'y, lg0, _ = (epol if epol is not None else pol).act(x, f.aux, teacher=st["T_now"])',
                   'st["mix"] = cfg["MIXED_ENCODE"](x)\n'
                   '                    y, lg0 = cfg["MIXED_DECODE"](x, f, st["mix"])')
    start = search.index("    def _search(obs, x, y, acts, xo, left, t0, tw):")
    base, target = search[:start], search[start:]
    target = patch(target, next(line for line in target.splitlines(True) if "tok, ctx = pol.enc_for" in line), "")
    fstart = target.index("        B0 = len(forced)")
    fend = target.index("        cand = [acts];", fstart)
    target = target[:fstart]+('        ys = [cfg["MIXED_DECODE"](x, f, st["mix"], v)[0] for v in forced]\n')+target[fend:]
    target = patch(target, "        for yy, mm in zip(ys, meta):\n            k = tuple(yy)",
                   "        for yy, mm in zip(ys, meta):\n            if yy is None:\n"
                   "                continue\n            k = tuple(yy)")
    src["cnet2_search"] = base + target
    head, rest = text.split("_SRC = ", 1)
    _, rest = rest.split("\n", 1)
    rest = rest.split("CONF = ", 1)[0]
    blocks = [head, "_SRC = "+repr(src)+"\n", rest, "CONF = "+repr(cfg)+"\n"]
    for dest, origin in (("_B1","_B1"),("_M1","_M1"),("_Q1","_Q1"),
                         ("_B2","_W64"),("_M2","_WMETA"),("_Q2","_QMETA")):
        blocks.append(dest+" = "+repr(hybrid[origin])+"\n")
    blocks.append(text[text.index("def _dequantize("):text.index("\n_W = ")])
    blocks.append((HERE/"d1_sapling_runtime.py").read_text())
    result = "\n".join(blocks).encode()
    compile(result, str(output), "exec")
    if len(result) > 10_000_000:
        raise ValueError("package too large")
    if output.exists() and output.read_bytes() != result:
        raise FileExistsError(output)
    if not output.exists():
        with output.open("xb") as file:
            file.write(result)
    print(len(result), hashlib.sha256(result).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    build(parser.parse_args().out)

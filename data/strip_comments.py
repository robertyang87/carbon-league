"""strip_comments IN OUT -- drop all comments and docstrings from a packed submission (outer file and every
embedded module source in _SRC). Uses ast round-trips, so semantics are unchanged; formatting is regenerated."""
import ast, sys

def strip(src):
    t = ast.parse(src)
    for n in ast.walk(t):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            b = n.body[0]
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    return t

src = open(sys.argv[1]).read()
t = strip(src)
for n in ast.walk(t):
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_SRC":
        d = ast.literal_eval(n.value)
        d = {k: ast.unparse(strip(v)) + "\n" for k, v in d.items()}
        n.value = ast.parse(repr(d), mode="eval").body
out = ast.unparse(t) + "\n"
open(sys.argv[2], "w").write(out)
print("bytes", len(src), "->", len(out), "| '#' lines left:", sum(1 for l in out.splitlines() if l.lstrip().startswith("#")))

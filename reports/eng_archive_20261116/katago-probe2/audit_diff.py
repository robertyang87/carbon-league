#!/usr/bin/env python3
"""Self-test 1: diff audit A/B vs original + weight-payload identity."""
import ast, base64, difflib, hashlib, zlib

P = {
    'orig': '/home/hatch/workspace/katago-probe/impl/echo3_ref.py',
    'A': '/home/hatch/workspace/katago-probe2/modules/echo3_search_pack.py',
    'B': '/home/hatch/workspace/katago-probe2/modules/echo3_sims_pack.py',
}
T = {k: open(v).read() for k, v in P.items()}
L = {k: v.split('\n') for k, v in T.items()}

def payloads(text):
    """all module-level string/bytes payloads by variable name, decoded bytes where b64"""
    tree = ast.parse(text)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == '_SRC':
                continue
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(val, str):
                out[name] = val
    return out

for other in ('A', 'B'):
    print(f'==== file-level diff orig -> {other} (changed line numbers) ====')
    sm = difflib.SequenceMatcher(None, L['orig'], L[other], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            print(f'{tag} orig[{i1+1}..{i2}] -> {other}[{j1+1}..{j2}]')
            for i in range(i1, i2):
                print(f'  - L{i+1} len={len(L["orig"][i])} head={L["orig"][i][:110]!r}')
            for j in range(j1, j2):
                print(f'  + L{j+1} len={len(L[other][j])} head={L[other][j][:110]!r}')
    # payload identity (weights etc.), excluding _SRC
    po, px = payloads(T['orig']), payloads(T[other])
    assert set(po) == set(px), (set(po) ^ set(px))
    for k in sorted(po):
        same = po[k] == px[k]
        print(f'payload {k}: identical={same} len={len(po[k])} md5={hashlib.md5(po[k].encode()).hexdigest()[:12]}')
        assert same, k
    # decoded weight bytes identity for _B1
    w_o = zlib.decompress(base64.b64decode(po['_B1']))
    w_x = zlib.decompress(base64.b64decode(px['_B1']))
    print(f'decoded _B1 weights: {len(w_o)} B identical={w_o == w_x} md5={hashlib.md5(w_o).hexdigest()}')

print('==== embedded cnet2_search decoded diff orig -> B ====')
def get_src(text):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == '_SRC':
            return ast.literal_eval(node.value)
so, sa, sb = get_src(T['orig']), get_src(T['A']), get_src(T['B'])
for k in so:
    print(f'module {k}: orig==A {so[k]==sa[k]}  orig==B {so[k]==sb[k]}')
d = difflib.unified_diff(so['cnet2_search'].split('\n'), sb['cnet2_search'].split('\n'),
                         'orig/cnet2_search', 'B/cnet2_search', lineterm='', n=1)
print('\n'.join(d))

#!/usr/bin/env python3
"""LP3 instrumenter: build the driver-side instrumented copy of the pinned
fixed-budget pack (gen3_fixed_u.py, md5 8a951aba7caebb18fd5bda65855aaadd).

Adds ONLY measurement plumbing; the search itself is untouched:
  P0  module-global _LP3 dict (vu / out / actual / actual_idx).
  P1  candidate injection: if _LP3['actual'] is set, the actual played action
      is added as a candidate right after act0/act_greedy, so it is valued in
      the SAME search tree under the SAME budget; its index is recorded.
  P2  u-scale capture: at look()'s two VALUE-SCALE BRIDGE sites the raw
      (pre-bridge) value-head outputs are kept in a parallel list; the val
      dict gains a twin dict `valu` filled at exactly the sites that fill
      `val` (main staged loop, terminal result() boards, top-2 verification),
      so valu[(c, m)] is the u-scale readout of the same leaf evaluation
      whose bridged _terminal-scale value is val[(c, m)]. For terminal
      boards the u readout is _u_of_margin(true margin) (the pack's own U).
  P3  export: after the final pick, _LP3['out'] receives best/actual_idx,
      copies of val/valu, depth, rolls, alive, n_cands.

Every patch is located by stripped-line anchors with exact occurrence-count
assertions; any drift aborts the build. Usage:
  python3 instrument_pack.py <pack.py> <out.py>
"""
import hashlib
import sys

EXPECTED_PACK_MD5 = '8a951aba7caebb18fd5bda65855aaadd'


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Patcher:
    def __init__(self, lines):
        self.lines = lines
        self.applied = []

    def find_unique(self, strip_text):
        idx = [i for i, ln in enumerate(self.lines) if ln.strip() == strip_text]
        assert len(idx) == 1, f'anchor {strip_text!r}: {len(idx)} occurrences'
        return idx[0]

    def indent_of(self, i):
        ln = self.lines[i]
        return ln[:len(ln) - len(ln.lstrip())]

    def insert_after(self, i, new_lines):
        ind = self.indent_of(i)
        self.lines[i + 1:i + 1] = [ind + t if t else t for t in new_lines]

    def replace_line(self, i, new_text):
        self.lines[i] = self.indent_of(i) + new_text

    def note(self, name):
        self.applied.append(name)


def build(pack_path, out_path):
    got = md5_of(pack_path)
    assert got == EXPECTED_PACK_MD5, f'pack md5 {got} != pinned {EXPECTED_PACK_MD5}'
    with open(pack_path) as f:
        lines = f.read().split('\n')
    p = Patcher(lines)

    # P0: globals after _ST = {}
    i = p.find_unique('_ST = {}')
    p.insert_after(i, ["_LP3 = {'vu': [], 'out': None, 'actual': None, 'actual_idx': -1}"])
    p.note('P0 globals')

    # P1: candidate injection after add_cand(act_greedy)
    i = p.find_unique('add_cand(act_greedy)')
    ind = p.indent_of(i)
    p.lines[i + 1:i + 1] = [
        ind + "if _LP3['actual'] is not None:",
        ind + "    add_cand(_LP3['actual'])",
        ind + "    _LP3['actual_idx'] = next(j for j, a in enumerate(cands) if a == _LP3['actual'])",
    ]
    p.note('P1 candidate injection')

    # P2a: look() init
    i = p.find_unique('def look(live, both, v4_k=1):')
    p.lines[i + 1:i + 1] = [p.indent_of(i) + '    _lp3_vu = []']
    p.note('P2a look init')

    # P2b: v4 net site -> stash raw u before bridging
    i = p.find_unique('_, _, vv = net(X, _SID, S)')
    p.insert_after(i, ['_lp3_raw = vv'])
    p.note('P2b v4 raw stash')

    # P2c: v4 append -> parallel u append (mean of raw u over the TTA group)
    i = p.find_unique('v.append(float(vv[v4_k * i:v4_k * i + v4_k].mean()))')
    p.replace_line(i, 'v.append(float(vv[v4_k * i:v4_k * i + v4_k].mean())); '
                      '_lp3_vu.append(float(_lp3_raw[v4_k * i:v4_k * i + v4_k].mean()))')
    p.note('P2c v4 u append')

    # P2d: v4 return -> publish vu
    i = p.find_unique('return v, []')
    p.lines[i:i] = [p.indent_of(i) + "_LP3['vu'] = _lp3_vu"]
    p.note('P2d v4 vu publish')

    # P2e: main net site -> stash raw u
    i = p.find_unique('lw, lb, vv = net(X, _SID, S)')
    p.insert_after(i, ['_lp3_raw = vv'])
    p.note('P2e main raw stash')

    # P2f: main extend -> parallel u extend (my-side entries are [::w])
    i = p.find_unique('v.extend(float(x) for x in vv); passes += 1')
    p.replace_line(i, 'v.extend(float(x) for x in vv); '
                      '_lp3_vu.extend(float(x) for x in _lp3_raw[::w]); passes += 1')
    p.note('P2f main u extend')

    # P2g: main return -> publish vu
    i = p.find_unique('return v, pol')
    p.lines[i:i] = [p.indent_of(i) + "_LP3['vu'] = _lp3_vu"]
    p.note('P2g main vu publish')

    # P2h: valu twin declaration
    i = p.find_unique('val = {}')
    p.replace_line(i, 'val = {}; valu = {}')
    p.note('P2h valu decl')

    # P2i: main staged-loop val assignment (the for-line whose next line is the val store)
    done = False
    for i, ln in enumerate(p.lines):
        if ln.strip() == 'for b, x in zip(live, v):' and \
                p.lines[i + 1].strip() == 'val[b[0], b[1]] = x':
            p.replace_line(i, 'for _lp3_i, (b, x) in enumerate(zip(live, v)):')
            p.replace_line(i + 1, "val[b[0], b[1]] = x; valu[b[0], b[1]] = _LP3['vu'][_lp3_i]")
            done = True
            break
    assert done, 'main val loop not found'
    p.note('P2i main val twin')

    # P2j: terminal boards in the main loop
    i = p.find_unique('val[b[0], b[1]] = result(b)')
    p.replace_line(i, 'val[b[0], b[1]] = result(b); valu[b[0], b[1]] = '
                      "_u_of_margin(float(b[2]['players'][me][0]) - float(b[2]['players'][op][0]))")
    p.note('P2j terminal twin (main)')

    # P2k: verification got declaration
    idx = [i for i, ln in enumerate(p.lines) if 'got = {}' in ln]
    assert len(idx) == 1, f'got decl: {len(idx)}'
    p.lines[idx[0]] = p.lines[idx[0]].replace('got = {}', 'got = {}; gotu = {}')
    p.note('P2k gotu decl')

    # P2l: verification final-look got assignment
    done = False
    for i, ln in enumerate(p.lines):
        if ln.strip() == 'for b, x in zip(live, v):' and \
                p.lines[i + 1].strip() == 'got[b[0], b[1]] = x':
            p.replace_line(i, 'for _lp3_i, (b, x) in enumerate(zip(live, v)):')
            p.replace_line(i + 1, "got[b[0], b[1]] = x; gotu[b[0], b[1]] = _LP3['vu'][_lp3_i]")
            done = True
            break
    assert done, 'verification got loop not found'
    p.note('P2l got twin')

    # P2m: verification val assignment
    i = p.find_unique('val[b[0], b[1]] = result(b) if over(b) else got[b[0], b[1]]')
    p.replace_line(i, 'val[b[0], b[1]] = result(b) if over(b) else got[b[0], b[1]]; '
                      'valu[b[0], b[1]] = _u_of_margin(float(b[2][\'players\'][me][0]) - '
                      "float(b[2]['players'][op][0])) if over(b) else gotu[b[0], b[1]]")
    p.note('P2m terminal twin (verification)')

    # P3: export after the final pick
    i = p.find_unique('act0 = cands[best]')
    p.insert_after(i, ["_LP3['out'] = {'best': best, 'actual_idx': _LP3['actual_idx'], "
                       "'val': dict(val), 'valu': dict(valu), 'depth': depth, 'rolls': rolls, "
                       "'alive': list(alive), 'n_cands': len(cands)}"])
    p.note('P3 export')

    text = '\n'.join(p.lines)
    compile(text, out_path, 'exec')  # syntax gate before writing
    with open(out_path, 'w') as f:
        f.write(text)
    print(f'instrumented pack written: {out_path}')
    print(f'patches applied ({len(p.applied)}): ' + ', '.join(p.applied))
    print(f'input md5 {got}; output md5 {md5_of(out_path)}')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])

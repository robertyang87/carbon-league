"""Make a standalone localized mix from unchanged parent payloads and the prior model's exact FP32 head."""
import ast
import base64
import hashlib
import json
import numpy as np
from preregister import HERE, PREV, PARENT, read, save, sha


RUNTIME = '''
def _local_candidates(x, aux, t):
    """Physical state support of the actually sampled collector on both training windows."""
    eligible = []
    if t not in _LOCAL_WINDOWS:
        return eligible
    for slot in np.flatnonzero(x["tok_mask"][:10]):
        if int(x["tok_kind"][slot]) != 0:
            continue
        pos = int(x["tok_pos"][slot])
        load = float(x["tok_cargo"][slot])
        distance = float(aux["dob"][pos])
        threat = int(aux["en_min"][pos] < load)
        limits = _LOCAL_RANGES.get((t, int(load >= 40.), threat))
        if limits is not None and limits[0] <= load <= limits[1] and limits[2] <= distance <= limits[3]:
            eligible.append(int(slot))
    return eligible


def _local_slot(x, aux, enc, t):
    eligible = _local_candidates(x, aux, t)
    if not eligible:
        return -1
    # Measure the parent return-adjusted distribution before applying any update.
    entropies = []
    _return_decode(x, aux, enc, True, entropy=entropies)
    choices = [(h, -slot) for slot, h in entropies if slot in eligible]
    return -max(choices)[1] if choices else -1

'''


def exactly(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one frozen anchor for {old[:80]!r}')
    return text.replace(old, new)


def build():
    plan = read(HERE / 'protocol.json')
    weights = PREV/'candidate.npz'
    assert sha(PARENT) == plan['references']['parent']
    assert sha(weights) == plan['references']['weights']
    assert sha(PREV/'candidate.py') == plan['references']['ungated']
    assert sha(PREV/'development/comparison.json') == plan['references']['historical_comparison']
    with np.load(weights) as npz:
        wt = np.ascontiguousarray(npz['whead.weight'].T,dtype='<f4')
        bias = np.ascontiguousarray(npz['whead.bias'],dtype='<f4')
    assert wt.shape == (96,5) and bias.shape == (5,)
    ranges = {tuple(map(int,k.split('/'))):tuple(v['cargo']+v['distance']) for k,v in plan['ranges'].items()}
    def embedded(a):
        b64=base64.b64encode(a.tobytes()).decode()
        return f'np.frombuffer(base64.b64decode({b64!r}), dtype="<f4").reshape({a.shape!r})'
    constants = f'\n_LOCAL_WINDOWS = {tuple(plan["windows"])!r}\n_LOCAL_RANGES = {ranges!r}\n_LOCAL_WT = {embedded(wt)}\n_LOCAL_B = {embedded(bias)}\n'
    src=PARENT.read_text()
    start=src.index('def _return_decode(')
    frozen_start, suffix = src[:start],src[start:]
    suffix=exactly(suffix,'def _return_decode(x, aux, encoded, apply_residual):',
                   'def _return_decode(x, aux, encoded, apply_residual, focal=-1, entropy=None):')
    suffix=exactly(suffix,'base = policy.lin(hidden, "whead")',
        'if slot == focal and policy is _NETS[1]:\n'
        '                    base = hidden @ _LOCAL_WT + _LOCAL_B\n'
        '                else:\n'
        '                    base = policy.lin(hidden, "whead")')
    old='        action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    suf='''        if entropy is not None and slot < 10 and int(x["tok_kind"][slot]) == 0 and np.count_nonzero(legal) >= 2:
            z = np.where(legal, combined, -np.inf)
            p = np.exp((z - np.max(z)).astype(np.float64))
            p /= p.sum()
            nz = p[p > 0]
            entropy.append((slot, float(-nz @ np.log(nz))))
'''
    suffix=exactly(suffix,old,suf+old)
    suffix=exactly(suffix,'    baseline = _return_decode(x, aux, enc, False)',
                   '    t = 299 - int(x["left"])\n    focal = _local_slot(x, aux, enc, t) if t in _LOCAL_WINDOWS else -1\n'
                   '    baseline = _return_decode(x, aux, enc, False, focal=focal)')
    suffix=exactly(suffix,'    candidate = _return_decode(x, aux, enc, True)',
                   '    candidate = _return_decode(x, aux, enc, True, focal=focal)')
    body=(frozen_start+constants+RUNTIME+suffix).encode()
    compile(body,'localized_sapling','exec')
    pack=HERE/'candidate.py'
    save(pack,body)
    save(HERE/'release.json',dict(candidate_sha256=sha(pack),parent_sha256=sha(PARENT),
         ungated_candidate_sha256=sha(PREV/'candidate.py'),protocol_sha256=sha(HERE/'protocol.json'),
         trained_checkpoint_sha256=sha(weights),trained_checkpoint_path='../balanced_return_dev/candidate.npz',
         changed_head_weight_transposed_sha256=hashlib.sha256(wt.tobytes()).hexdigest(),
         changed_head_bias_sha256=hashlib.sha256(bias.tobytes()).hexdigest(),
         frozen_parent_frontmatter_sha256=hashlib.sha256(frozen_start.encode()).hexdigest(),
         builder_sha256=sha(HERE/'build.py'),parent_frontmatter_bytes=len(frozen_start.encode()),bytes=len(body)))


if __name__=='__main__':
    build()

"""Build only when frozen paired validation qualifies; verify all insertion sites on the original bytes."""
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEARN = HERE.parent / 'return_response_learning'
PARENT = HERE.parent / 'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def export_model():
    lines = (LEARN/'model.py').read_text().splitlines()
    replacements = {'_RR_DIST = ': 'FEAT_DIST', '_RR_MOVE = ': '_OPP_STEPS'}
    for prefix, symbol in replacements.items():
        indices = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if len(indices) != 1:
            raise ValueError('Missing unique geometry table ' + prefix)
        lines[indices[0]] = prefix + symbol
    return '\n'.join(lines) + '\n'


def insertions(head, scale):
    layer = export_model() + '\n' + (HERE/'selector.py').read_text()
    layer += f'\n_RR_MODEL = {head!r}\n_RR_SCALE = {scale!r}\n'
    layer += '\ndef _rr_bound_scores(players, me, step, commands):\n    return rr_scores(players, me, step, commands, _RR_MODEL)\n\n'
    return [
        ('def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):\n', False, layer),
        ("    players = obs['players']\n    full = [1.0] * N\n", True,
         "    rr_active = step in (279, 285)\n"
         "    if rr_active:\n"
         "        rr_hire = min(float(configuration.get('recCollectorCost', 30)), float(configuration.get('recPlanterCost', 30)))\n"),
        ('                out.append([c, m, _turn(root, configuration, acts), players])\n', True,
         "                if rr_active:\n                    out[-1].append(_rr_in_play(out[-1][2]['players'], rr_hire))\n"),
        ("            b[3] = o['players']; b[2] = _turn(o, configuration, [am, ao] if me == 0 else [ao, am])\n", True,
         "            if rr_active:\n                b[4] = b[4] and _rr_in_play(b[2]['players'], rr_hire)\n"),
        ("    if _S_REC is not None:\n        _S_REC.append((step, me, cands,", False,
         "    if rr_active:\n"
         "        best, rr_reason, rr_detail = _rr_select(best, alive, mean, dep, run, cands, step, me, players,\n"
         "            depth, rolls, cut, t0, _time.perf_counter, _rr_bound_scores, _RR_SCALE, _S_EPS)\n"),
    ]


def anchored_layout(raw, edits):
    if sha(raw) != PARENT_SHA:
        raise ValueError('Parent differs from the live v2')
    encode = lambda s: s.replace('\n', '\r\n').encode()
    parts = []
    cursor = 0
    for anchor, after, body in edits:
        marker = encode(anchor)
        if raw.count(marker) != 1:
            raise ValueError('Nonunique parent anchor: ' + anchor)
        at = raw.index(marker) + (len(marker) if after else 0)
        if at <= cursor:
            raise ValueError('Overlapping anchors')
        parts += [(False, raw[cursor:at]), (True, encode(body))]
        cursor = at
    parts.append((False, raw[cursor:]))
    if b''.join(chunk for added, chunk in parts if not added) != raw:
        raise ValueError('Original bytes failed reconstruction')
    return parts


def main():
    wfile, hfile, pfile, ffile = (LEARN/n for n in ('weights.json','held.json','protocol.json','fit.json'))
    weights, held, fit = read(wfile), read(hfile), read(ffile)
    identity = sha(wfile.read_bytes())
    if not (held['weight_sha'] == identity == fit['weights'] and fit['protocol'] == sha(pfile.read_bytes())):
        raise ValueError('Assessment, weights, or protocol identity mismatch')
    if (weights['depth'], weights['steps'], weights['receipt_window'], weights['input_width']) != (8,[279,285],12,21):
        raise ValueError('Unrecognized model support')
    scale = weights['pair_scale']
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('Invalid cash-to-critic scale')
    parts = anchored_layout(PARENT.read_bytes(), insertions({'enhanced': weights['enhanced']}, scale))
    raw = b''.join(chunk for _, chunk in parts)
    compile(raw, 'in_memory_source_layout', 'exec')
    checks = held['checks']
    required = {'ordinary_brier_better_than_both', 'active_three','actual_margin_gain','correct_two','four_hold_roots','no_new_wrong','paired_mse_improves','scale_positive'}
    qualifies = held['qualified_for_development'] is True and set(checks)==required and all(v is True for v in checks.values()) and scale>0
    out = HERE/'submission.py'
    manifest = dict(parent_sha256=PARENT_SHA,weights_sha256=identity,training_protocol_sha256=fit['protocol'],
                    model_sha256=sha((LEARN/'model.py').read_bytes()),selector_sha256=sha((HERE/'selector.py').read_bytes()),
                    held_sha256=sha(hfile.read_bytes()),anchored_insertions=len(parts)//2,
                    stripped_matches_parent=True,source_compiles=True,qualified=qualifies,
                    failed_checks=sorted(k for k,v in checks.items() if v is not True),published_in_this_run=False)
    if qualifies:
        tmp = out.with_suffix('.py.tmp')
        tmp.write_bytes(raw)
        os.replace(tmp, out)
        manifest.update(submission_sha256=sha(raw), published_in_this_run=True)
    manifest['submission_exists'] = out.exists()
    status = HERE/'build_status.json'
    status.write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(manifest,sort_keys=True))


if __name__ == '__main__':
    main()

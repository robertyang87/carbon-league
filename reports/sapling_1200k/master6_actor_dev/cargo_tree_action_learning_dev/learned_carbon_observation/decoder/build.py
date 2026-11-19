"""Expose one frozen estimate to Sapling's native move features using a separately advanced Prefix."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/"protocol.json").read_text())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(source, old, new):
    assert source.count(old)==1,old
    return source.replace(old,new)


def build(name,key):
    previous=(HERE.parent/(name+".py")).read_bytes()
    assert sha(previous)==CFG[key]
    source=previous.decode()
    # Preserve the original encoder rounding and supply continuous native-unit values to the decoder.
    source=replace_once(source,'def _carbon_view(x, feat, F, weights):','def _carbon_views(x, aux, feat, F, weights):')
    source=replace_once(source,'    if not eligible.any():\n        return x','    if not eligible.any():\n        return x, aux')
    source=replace_once(source,'''    corrected = F.q8(base + change)
    maps[0].reshape(-1)[eligible] = corrected[eligible]
    return dict(x, maps=maps)''','''    prediction = np.clip(base + change, 0, 1)
    maps[0].reshape(-1)[eligible] = F.q8(prediction)[eligible]
    est = aux["est"].copy()
    est[eligible] = F.f16(prediction[eligible])
    return dict(x, maps=maps), dict(aux, est=est)''')
    source=replace_once(source,'student_x = _carbon_view(x, _FEAT, F, _CARBON_WEIGHTS)',
                        'student_x, student_aux = _carbon_views(x, aux, _FEAT, F, _CARBON_WEIGHTS)')
    for arg in ('True','False'):
        source=replace_once(source,'_return_decode(x, aux, enc, '+arg+')','_return_decode(x, aux, enc, '+arg+', student_aux)')
    h=source.index('def _return_decode(')
    i=source.index('def _return_basis(',h)
    func=source[h:i]
    func=replace_once(func,'def _return_decode(x, aux, encoded, apply_residual):',
                      'def _return_decode(x, aux, encoded, apply_residual, student_aux):')
    func=replace_once(func,'    prefix = F.Prefix(x, aux)','    prefix = F.Prefix(x, aux)\n    student_prefix = F.Prefix(x, student_aux)')
    func=replace_once(func,'        unit, move, legal = prefix.feats(slot)',
                      '        unit, move, legal = prefix.feats(slot)\n        student_unit, student_move, _ = student_prefix.feats(slot)')
    func=replace_once(func,'        for state in states:\n            policy, tokens, context, hidden = state',
                      '''        for index, state in enumerate(states):
            policy, tokens, context, hidden = state
            policy_unit, policy_move = (student_unit, student_move) if index == 1 else (unit, move)''')
    func=replace_once(func,'[tokens[slot], context, unit, move.reshape(-1)]',
                      '[tokens[slot], context, policy_unit, policy_move.reshape(-1)]')
    func=replace_once(func,'policy.lin(move, "V")','policy.lin(policy_move, "V")')
    func=replace_once(func,'        prefix.advance(slot, action)','        prefix.advance(slot, action)\n        student_prefix.advance(slot, action)')
    out=(source[:h]+func+source[i:]).encode()
    assert len(out)<=10_000_000
    (HERE/(name+".py")).write_bytes(out)
    return dict(parent_sha256=sha(previous),sha256=sha(out),bytes=len(out))


if __name__=="__main__":
    assert sha((HERE.parent/"weights.npz").read_bytes())==CFG["weights"]
    arms={a:build(a,key) for a,key in (("candidate","encoder_student"),("constant","encoder_constant"))}
    (HERE/"packages.json").write_text(json.dumps(dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()),arms=arms),indent=2)+"\n")

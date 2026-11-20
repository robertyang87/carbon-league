"""Assemble just the root opponent heads from the exact online V2 bytes."""
import ast
import json
from register import HERE,BASE,sha


def assemble():
    p=json.loads((HERE/'protocol.json').read_bytes())
    w=json.loads((HERE/'weights.json').read_bytes())
    assert sha(BASE.read_bytes())==w['parent']==p['parent']
    assert w['protocol']==sha((HERE/'protocol.json').read_bytes())
    parent=BASE.read_bytes()
    raw=(HERE/'runtime.py').read_text()
    asts=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)]
    helpers="\n\n".join(ast.get_source_segment(raw,f) for f in asts)
    assert [f.name for f in asts]==['opponent_context','opponent_adjust']
    extra=("OPP_ADAPTATION = np.asarray("+repr(w['coeff'])+",dtype=np.float32)\n\n"+helpers+"\n\n").replace('\n','\r\n').encode()
    marker=b'def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):'
    site=b'    lwo = _feedback_adjust(lwo, players, op, _FEEDBACK_WEIGHTS, _OPP_STEPS, FEAT_DIST)\r\n'
    edit=(b'    context = opponent_context(players, op, step, _OPP_STEPS, FEAT_DIST)\r\n'
          b'    lwo, lbo = opponent_adjust(lwo, lbo, context, OPP_ADAPTATION)\r\n')
    assert all(parent.count(m)==1 for m in (marker,site)) and b'OPP_ADAPTATION' not in parent
    new=parent.replace(marker,extra+marker).replace(site,site+edit)
    assert new.replace(site+edit,site).replace(extra+marker,marker)==parent and len(new)<10_000_000
    compile(new,'verified.py','exec')
    return new

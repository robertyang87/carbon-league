"""Embed the unchanged public-feature cash head in the online v4 source."""
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parents[2]
PARENT=D/'advantage_policy_update/v2_bc.py'
PARENT_SHA='cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a'
FEAT=D/'ladder/tree_payback/runtime.py'
WEIGHTS=D/'tree_control_response/weights.json'
sha=lambda b:hashlib.sha256(b).hexdigest()


def build():
    raw=PARENT.read_bytes();feat=FEAT.read_bytes();wraw=WEIGHTS.read_bytes();W=json.loads(wraw)
    assert sha(raw)==PARENT_SHA and sha(feat)=='b6fb7ba5af74dfad8fd2dd72e9d788b849fc451812df644005444f2c166ebc56' and sha(wraw)=='133a3b77a76a1f160e2fc333feea538c49150f6094f54546783590583c047741'
    assert len(W['feature_mean'])==len(W['feature_scale'])==len(W['coef'])==23 and min(W['feature_scale'])>0
    # Retain the exact feature implementation without embedding the unrelated risk classifier.
    src=feat.decode().split('\ndef predict(feature_row, head):')[0]
    conf='_PLANT_LIB={}\nexec('+repr(src)+', _PLANT_LIB)\n'
    for symbol,arr in (('_PLANT_MEAN',W['means'][0]),('_PLANT_XMEAN',W['feature_mean']),('_PLANT_SCALE',W['feature_scale']),('_PLANT_COEF',[v[0] for v in W['coef']])):
        conf+=f'{symbol}={arr!r}\n'
    rt=(HERE/'runtime.py').read_bytes(); added=(conf+'\n'+rt.decode().rstrip()+'\n\n\n').replace('\n','\r\n').encode(); marker=b'def _act(obs, configuration=None, t0=None):'; site=b'    lw, lb = _POLICY.forward(x, _SID, own_root=True)\r\n';call=b"    _plant_reprice(obs, configuration, st['tr'], lw)\r\n"
    assert raw.count(marker)==raw.count(site)==1
    out=raw.replace(marker,added+marker).replace(site,site+call);assert out.replace(added,b'').replace(call,b'')==raw and len(out)<10_000_000
    compile(out,'submission.py','exec'); return out,dict(parent_sha256=sha(raw),candidate_sha256=sha(out),candidate_bytes=len(out),model_sha256=sha(wraw),feature_source_sha256=sha(feat),runtime_sha256=sha(rt))


if __name__=='__main__':
    raw,cert=build();(HERE/'submission.py').write_bytes(raw);(HERE/'identity.json').write_text(json.dumps(cert,indent=2)+'\n');print(cert)

"""Inject only the newly trained embedding into the acting seat's root forward."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
PARENT=FAMILY/'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA='8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    raw=PARENT.read_bytes()
    assert sha(raw)==PARENT_SHA
    weights=(HERE/'weights.json').read_bytes()
    audit=json.loads((HERE/'held.json').read_bytes())
    assert audit['weights']==sha(weights)
    maps=audit['maps'].values()
    improves=sum(m['worker_trained_nll']<m['worker_original_nll'] for m in maps)
    assert improves>len(maps)/2
    row=json.loads(weights)['embedding']
    assert len(row)==96
    source=raw.decode().replace('\r\n','\n')
    style="_SID = _POLICY.sid('TJB-o18-rush-v41')\n"
    forward='    lw, lb = _POLICY.forward(x, _SID)'
    assert source.count(style)==source.count(forward)==1
    append=f"_OWN_STYLE = len(_POLICY.p['emb'])\n_POLICY.p['emb'] = np.vstack((_POLICY.p['emb'], np.asarray({row!r}, dtype=np.float32)))\n"
    result=source.replace(style,style+append).replace(forward,forward.replace('_SID','_OWN_STYLE'))
    result=result.replace('\n','\r\n').encode()
    dest=HERE/'candidate.py'
    compile(result,str(dest),'exec')
    if dest.exists():
        assert dest.read_bytes()==result
    else:
        dest.write_bytes(result)
    return sha(result)


if __name__=='__main__':
    print(build())

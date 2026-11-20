"""Use the fitted part of the bilateral cash prediction that total cargo cannot see."""
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
    weight_raw=(HERE/'weights.json').read_bytes()
    weights=json.loads(weight_raw)['coefficients']
    evaluation=json.loads((HERE/'held.json').read_bytes())
    assert evaluation['weights_sha256']==sha(weight_raw)
    maps=evaluation['maps'].values()
    assert sum(m['models']['routes']['mae']<m['models']['totals']['mae'] for m in maps)>len(maps)/2
    feature=(HERE/'features.py').read_text()
    feature=feature[feature.index('NAMES = '):].replace('DISTANCE','_DEPOSIT_DISTANCE').replace('MOVES','_DEPOSIT_MOVES')
    feature=feature.replace('def raw(', 'def _deposit_features(').replace('NAMES','_DEPOSIT_NAMES')
    weight_prefix=f"_DEPOSIT_MODEL = {weights!r}\n"
    runtime='''
def _deposit_residual(obs, me):
    own = _deposit_features(obs['players'], me, obs['step'])
    foe = _deposit_features(obs['players'], 1-me, obs['step'])
    x = [a-b for a,b in zip(own, foe)]
    routes = sum(w*v for w,v in zip(_DEPOSIT_MODEL['routes'], x))
    totals = sum(w*v for w,v in zip(_DEPOSIT_MODEL['totals'], x))
    # Keep the critic in charge. The model sees collectible cash only.
    return float(np.clip((routes-totals)/6000.0, -.09, .09))


'''
    a=raw.index(b'def _search(')
    src=raw[a:].decode().replace('\r\n','\n')
    old = '''            for b, x in zip(live, v):
                val[b[0], b[1]] = x'''
    new = '''            for b, x in zip(live, v):
                if t == D2 and 40 <= b[2]['step'] <= 289:
                    x += _deposit_residual(b[2], me)
                val[b[0], b[1]] = x'''
    assert src.count(old)==1
    destination=HERE/'candidate.py'
    result=raw[:a]+(feature+'\n'+weight_prefix+runtime+src.replace(old,new)).replace('\n','\r\n').encode()
    compile(result,str(destination),'exec')
    if destination.exists():
        assert destination.read_bytes()==result
    else:
        destination.write_bytes(result)
    return sha(result)


if __name__=='__main__':
    print(build())

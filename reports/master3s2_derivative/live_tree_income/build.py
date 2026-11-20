"""Build a V2 candidate that uses only the learned correction to a duration-only rent estimate."""
import json

from train import FAMILY, HERE, PARENT, sha


def build():
    parent = FAMILY/'ladder/prediction_feedback/package/candidate.py'
    raw = parent.read_bytes()
    assert sha(raw) == PARENT
    weights_file = (HERE/'weights.json').read_bytes()
    weights = json.loads(weights_file)
    held = json.loads((HERE/'held.json').read_bytes())
    assert held['weights_sha256'] == sha(weights_file)
    maps = held['maps'].values()
    assert sum(m['margin_mae']['prediction'] < m['margin_mae']['remaining_turn_baseline'] for m in maps) > len(maps)/2
    # Remove the module docstring and use the bot's public state primitives directly.
    feature_code = (HERE/'features.py').read_text()
    feature_code = feature_code[feature_code.index('HORIZON = 16'):]
    feature_code = feature_code.replace('def extract(bot,', 'def _existing_tree_features(').replace('bot.', '')
    feature_code = feature_code.replace('HORIZON', '_RENT_HORIZON')
    params = f"_RENT_FIT = {weights!r}\n"
    correction = '''
def _rent_residual(ob, me, known):
    data = [(owner, values) for _, owner, values in _existing_tree_features(
        ob['players'], ob['step'], ob['carbon'], known)]
    if not data:
        return 0.0
    x = np.asarray([values for _, values in data])
    p = np.maximum(0, _RENT_FIT['intercept'] + ((x-_RENT_FIT['mean'])/_RENT_FIT['scale']) @ _RENT_FIT['coef'])
    residual = p - _RENT_FIT['simple_per_remaining_turn'] * x[:, 1]
    signed = sum(float(v) if owner == me else -float(v) for (owner, _), v in zip(data, residual))
    # Keep the original critic and add one quarter of the fitted cash residual.
    return float(np.clip(signed / 12000.0, -.06, .06))

'''
    first = raw.index(b'def _search(')
    end = raw.index(b'\r\n_net()', first)
    code = raw[first:end].decode().replace('\r\n','\n')
    old = '''            for b, x in zip(live, v):
                val[b[0], b[1]] = x'''
    new = '''            for b, x in zip(live, v):
                if t == D2 and 20 <= step <= 92:
                    x += _rent_residual(b[2], me, _ST['tr'].bel.known)
                val[b[0], b[1]] = x'''
    assert code.count(old) == 1
    code = code.replace(old, new)
    result = raw[:first]+(feature_code+'\n'+params+correction+code).replace('\n','\r\n').encode()+raw[end:]
    target = HERE/'candidate.py'
    compile(result, str(target), 'exec')
    if target.exists():
        assert target.read_bytes() == result
    else:
        target.write_bytes(result)
    return sha(result)


if __name__ == '__main__':
    print(build())

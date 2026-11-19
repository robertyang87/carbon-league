from pathlib import Path
from hashlib import sha256
HERE=Path(__file__).resolve().parent
def build():
    parent=HERE.parent/'planter_capture_full_dev/candidate.py';text=parent.read_text()
    assert sha256(parent.read_bytes()).hexdigest()=='543a9d927ec9a2f6b6e2020ac2236eb9aecb594c0ee16a44c1a0ddee86f23797'
    old='    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))';assert text.count(old)==1
    text=text.replace(old,'    return _role_orders(observation, configuration, _FEAT.to_actions(_mixed_actions(x, _FEAT.aux)))')
    geom=(HERE.parent/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
    text+='\n'+geom+'\ndef _role_choose('+(HERE/'role.py').read_text().split('def choose(',1)[1]+'''
_ROLE_USED = {}
def _role_orders(obs, cfg, orders):
    seat = obs['player']
    if obs['step'] == 0:
        _ROLE_USED[seat] = False
    if 120 <= obs['step'] <= 200 and not _ROLE_USED.get(seat):
        move = _role_choose(obs, orders, cfg)
        if move:
            orders[move['worker']] = move['alternative']
            _ROLE_USED[seat] = True
    return orders
'''
    (HERE/'candidate.py').write_text(text)
if __name__=='__main__':build()

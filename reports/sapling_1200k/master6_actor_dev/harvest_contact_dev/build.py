"""Produce a standalone, one-intervention-per-game candidate from the exact source parent."""
from pathlib import Path
from hashlib import sha256
HERE=Path(__file__).resolve().parent
def build():
    parent=HERE.parent/'planter_capture_full_dev/candidate.py'
    assert sha256(parent.read_bytes()).hexdigest()=='543a9d927ec9a2f6b6e2020ac2236eb9aecb594c0ee16a44c1a0ddee86f23797'
    text=parent.read_text();call='    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))'
    assert text.count(call)==1
    geom=(HERE.parent/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
    choose=(HERE/'contact.py').read_text().split('def choose(',1)[1]
    harvest=(HERE/'occupy.py').read_text().split('def harvest(',1)[1]
    text=text.replace(call,'    return _patch_orders(observation, _FEAT.to_actions(_mixed_actions(x, _FEAT.aux)))')
    text+='\n'+geom+'\ndef _patch_choose('+choose+'\ndef _patch_harvest('+harvest+'''
_PATCH_HISTORY = {}
_PATCH_ROOT = {}
def _patch_orders(obs, orders):
    seat = obs['player']
    t = obs['step']
    if t == 0:
        _PATCH_HISTORY.pop(seat, None)
        _PATCH_ROOT.pop(seat, None)
    previous = _PATCH_HISTORY.get(seat)
    _PATCH_HISTORY[seat] = {'step': t, 'players': obs['players']}
    root = _PATCH_ROOT.get(seat)
    if root:
        if t == root['t'] + 1:
            return _patch_harvest(obs, orders, root)
    elif 120 <= t <= 200:
        root = _patch_choose(obs, orders, previous)
        if root:
            root['site'] = obs['players'][1-seat][2][root['target']][0]
            _PATCH_ROOT[seat] = root
            orders[root['worker']] = root['alternative']
    return orders
'''
    (HERE/'candidate.py').write_text(text)
if __name__=='__main__':build()

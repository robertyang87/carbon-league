from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
p=HERE.parent/'planter_capture_full_dev/candidate.py'
src=p.read_text()
assert hashlib.sha256(p.read_bytes()).hexdigest()=='543a9d927ec9a2f6b6e2020ac2236eb9aecb594c0ee16a44c1a0ddee86f23797'
needle='    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))'
assert src.count(needle)==1
geo=(HERE.parent/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
route=(HERE/'rescue.py').read_text().split('def choose(',1)[1]
src=src.replace(needle,'''    orders = _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))
    return _deadline_route(observation, orders)''')
src+='\n'+geo+'\ndef _rescue_choice('+route+'''
_DEADLINE_FOCUS = {}
def _deadline_route(observation, orders):
    seat = observation['player']
    if observation['step'] == 0:
        _DEADLINE_FOCUS.pop(seat, None)
    if observation['step'] < 280:
        return orders
    move = _rescue_choice(observation, orders, _DEADLINE_FOCUS.get(seat))
    if move:
        _DEADLINE_FOCUS[seat] = move['worker']
        orders[move['worker']] = move['alternative']
    return orders
'''
(HERE/'candidate.py').write_text(src)

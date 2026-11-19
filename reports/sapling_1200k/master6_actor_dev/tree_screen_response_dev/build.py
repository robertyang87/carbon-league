"""Build the registered one-step guard around the unchanged Sapling actor and joint decoder."""
from hashlib import sha256
from pathlib import Path

HERE=Path(__file__).resolve().parent

def build():
    actor=HERE.parent/'planter_capture_full_dev/candidate.py'
    assert sha256(actor.read_bytes()).hexdigest()=='543a9d927ec9a2f6b6e2020ac2236eb9aecb594c0ee16a44c1a0ddee86f23797'
    source=actor.read_text()
    entry='    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))'
    assert source.count(entry)==1
    source=source.replace(entry,'    return _screen_orders(observation, configuration, _FEAT.to_actions(_mixed_actions(x, _FEAT.aux)))')
    geometry=(HERE.parent/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
    choose=(HERE.parent/'tactical_exchange_dev/approach.py').read_text().split('def choose(',1)[1]
    wrapper="""
_SCREEN_LAST = {}
def _screen_orders(obs, cfg, commands):
    seat = obs['player']
    t = obs['step']
    last_t, enemies = _SCREEN_LAST.get(seat, (-2, {}))
    previous = enemies if t > 0 and last_t == t-1 else {}
    _SCREEN_LAST[seat] = (t, {uid: tuple(w) for uid,w in obs['players'][1-seat][2].items()})
    move = _screen_choose(obs, commands, cfg, previous)
    if move:
        commands[move['uid']] = move['after']
    return commands
"""
    packed=source+'\n'+geometry+'\ndef _screen_choose('+choose+wrapper
    (HERE/'candidate.py').write_text(packed)
    print(sha256(packed.encode()).hexdigest(),len(packed.encode()))
if __name__=='__main__':build()

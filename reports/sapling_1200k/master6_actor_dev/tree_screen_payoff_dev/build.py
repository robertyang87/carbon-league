"""Build a standalone research candidate with action-level economic pricing after Sapling's normal decoder."""
from hashlib import sha256
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent

def build():
    parent=ACTOR/'planter_capture_full_dev/candidate.py'
    assert sha256(parent.read_bytes()).hexdigest()=='543a9d927ec9a2f6b6e2020ac2236eb9aecb594c0ee16a44c1a0ddee86f23797'
    text=parent.read_text()
    entry='    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))'
    assert text.count(entry)==1
    text=text.replace(entry,'    return _economy_screen(observation, configuration, _FEAT.to_actions(_mixed_actions(x, _FEAT.aux)))')
    geometry=(ACTOR/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
    choose=(ACTOR/'tactical_exchange_dev/approach.py').read_text().split('def choose(',1)[1]
    price=(HERE/'payoff.py').read_text().split('HORIZON=',1)[1]
    text+='\n'+geometry+'\nfrom math import exp\n\ndef _choose_screen('+choose+'\nHORIZON='+price
    text+='\n_ECON_WEIGHTS='+repr(json.loads((HERE/'weights.json').read_bytes()))
    text+='''
_ECON_HISTORY={}
def _economy_screen(obs,cfg,commands):
    seat,t=obs["player"],obs["step"]
    time,prev=_ECON_HISTORY.get(seat,(-2,{}))
    if time!=t-1 or not t:prev={}
    _ECON_HISTORY[seat]=(t,{uid:tuple(w) for uid,w in obs["players"][1-seat][2].items()})
    if t+2>299:return commands
    root=_choose_screen(obs,commands,cfg,prev)
    if root and price(obs,commands,cfg,root,prev,_ECON_WEIGHTS)["accept"]:
        commands[root["uid"]]=root["after"]
    return commands
'''
    (HERE/'candidate.py').write_text(text)
if __name__=='__main__':build()

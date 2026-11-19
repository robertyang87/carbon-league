"""Rebuild the standalone student-based policy with a freight routing override."""
from pathlib import Path
from hashlib import sha256
HERE=Path(__file__).resolve().parent
def build():
    parent=HERE.parent/'planter_capture_full_dev/candidate.py'
    text=parent.read_text()
    assert sha256(parent.read_bytes()).hexdigest()=='543a9d927ec9a2f6b6e2020ac2236eb9aecb594c0ee16a44c1a0ddee86f23797'
    old='    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))';assert text.count(old)==1
    geometry=(HERE.parent/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
    fn=(HERE/'routing.py').read_text().split('def choose(',1)[1]
    text=text.replace(old,'    return _toll_orders(observation,configuration,_FEAT.to_actions(_mixed_actions(x, _FEAT.aux)))')
    text+='\n'+geometry+'\ndef _toll_choose('+fn+'''
def _toll_orders(obs,cfg,orders):
    move=_toll_choose(obs,orders,cfg,_D,_TO,_DIRECTION)
    if move:orders[move['worker']]=move['alternative']
    return orders
'''
    (HERE/'candidate.py').write_text(text)
if __name__=='__main__':build()

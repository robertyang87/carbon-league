from pathlib import Path
from hashlib import sha256
import json
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent
def build():
    source=(ACTOR/'actor_partner_dev/candidate.py').read_bytes()
    assert sha256(source).hexdigest()=='c5ad3f43959a4812e84e89574ee0c6e69765ebb06a8b3ec93f839b0dde128dd3'
    geometry=(ACTOR/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
    rule=(HERE/'decision.py').read_text().split('FEATURES=',1)[1]
    code=source.decode()+'\n'+geometry+'\nFEATURES='+rule+'\n'+(HERE/'overlay.py').read_text()
    code+='\n_GROUND_WEIGHTS='+repr(json.loads((HERE/'weights.json').read_bytes())['coef'])
    code+='''
_GROUND_SELECTION=GroundSelection()
_GROUND_PARENT=agent
def agent(observation,configuration=None):
    orders=_GROUND_PARENT(observation,configuration)
    return _GROUND_SELECTION.act(observation,configuration,orders,_GROUND_WEIGHTS)
'''
    (HERE/'candidate.py').write_text(code)
if __name__=='__main__':build()

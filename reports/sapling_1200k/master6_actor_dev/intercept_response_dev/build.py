"""Add one fitted conditional-action forecast to the existing Sapling/expert decoder."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'planter_capture_full_dev/candidate.py'
def build():
    assert json.loads((HERE/'evaluation.json').read_bytes())['admitted']
    source=PARENT.read_text()
    hook='    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))'
    before='''    physical_cargo = cargo_map(observation, _CARRIER_WEIGHTS)
    F = sys.modules["cnet2_feat"]
    _FEAT.aux["opponent_cargo_flow"] = physical_cargo[F.CANON] if observation["player"] else physical_cargo
'''
    marker='        if apply_residual and slot < 10 and x["tok_kind"][slot] == 0:'
    rule='''        if slot < 10 and x["tok_kind"][slot] == 0 and x["tok_cargo"][slot] < 60:
            dest = F.DEST[int(x["tok_pos"][slot])]
            safe = (move[:, 0] == 0) & (move[:, 1] == 0) & (move[:, 6] == 0)
            combined = combined + 1.2 * safe * aux["opponent_cargo_flow"][dest]
'''
    assert source.count(hook)==source.count(marker)==1
    weights=np.load(HERE/'weights.npz')['weights'].tolist()
    cand=source.replace(hook,before+hook).replace(marker,rule+marker)+'\n'+(HERE/'forecast.py').read_text()+'\n_CARRIER_WEIGHTS=np.array('+repr(weights)+')\n'
    (HERE/'candidate.py').write_text(cand)
    prev=json.loads((HERE.parent/'capture_reach_dev/protocol.json').read_bytes())
    prev.update(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seeds=[621758324],
                candidate=hashlib.sha256(cand.encode()).hexdigest(),parent=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
                rule='Intercept with own load <60, threat-free non-enemy-base destinations, posterior cargo mass from public carrier model. Cap reward at 1.2 logits. Existing both-network inference and joint decoding retained.')
    (HERE/'protocol.json').write_text(json.dumps(prev,indent=2)+'\n')
if __name__=='__main__':build()

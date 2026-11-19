"""Build an uploadable single file from the frozen hybrid plus public bank decisions."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
parent=ACTOR/'planter_capture_full_dev/candidate.py'
src=parent.read_text()
needle='    return _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))'
assert src.count(needle)==1
geometry=(ACTOR/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
rule=(HERE/'bank.py').read_text().split('def choose(',1)[1]
extra=geometry+'\ndef _bank_choice('+rule
src=src.replace(needle,'''    orders = _FEAT.to_actions(_mixed_actions(x, _FEAT.aux))
    decision = _bank_choice(observation, orders)
    if decision:
        orders[decision["worker"]] = decision["alternative"]
    return orders''')
(HERE/'candidate.py').write_text(src+'\n'+extra)

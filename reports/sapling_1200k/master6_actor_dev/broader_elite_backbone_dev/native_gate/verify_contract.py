"""Read-back check of code, protocol and sole bound output, using the preweight receipt contract."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent; T=H.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
contract=read(H/'receipt_contract.json'); bind=read(H/'binding.json')
for key,path in [('protocol_sha256',H/'protocol.json'),('gate_source_sha256',H/'gate.py'),('match_source_sha256',H/'match.py'),('recipe_sha256',T/'recipe.json')]:
    assert contract[key]==sha(path),key
assert bind['protocol']==contract['protocol_sha256'] and bind['files']['recipe.json']==contract['recipe_sha256']
for _,p in contract['files'].items():
    f=(H/p).resolve();assert f.is_relative_to(T) and bind['files'][f.name]==sha(f)
print('Verified frozen gate, trainer recipe, original input and sole final checkpoint',bind['files']['candidate.npz'])

"""Build a one-file read-only economic forecast head with no runtime package/file dependencies."""
from pathlib import Path
import base64,gzip,json,hashlib,tempfile,importlib.util
from protocol import HERE,sha
def build():
    weights=HERE/"weights.json.gz"; raw=weights.read_bytes()
    obj=json.loads(gzip.decompress(raw))
    for k,p in (("training_protocol",HERE/"protocol.json"),("training_receipt",HERE/"train_sources.json")):
        if obj[k]!=sha(p):raise ValueError("training identity mismatch")
    added="""
import base64 as _base64, gzip as _gzip, json as _json
_WEIGHTS = _json.loads(_gzip.decompress(_base64.b85decode(%r)))
def forecast(observation, history, worker_uid, planned_action=None, friendly_orders=None):
    return predict(_WEIGHTS, observation, history, observation["player"], worker_uid, planned_action, friendly_orders)
"""%base64.b85encode(raw)
    packet=(HERE/"head.py").read_bytes()+added.encode()
    if len(packet)>1_000_000:raise ValueError("forecast module exceeds its size target")
    compiled=compile(packet,"lot_forecast.py","exec")
    namespace={};exec(compiled,namespace)
    if namespace["_WEIGHTS"]!=obj:raise ValueError("embedded weight mismatch")
    target=HERE/"lot_forecast.py"
    if target.exists() and target.read_bytes()!=packet:raise ValueError("would alter published weights or definition")
    target.write_bytes(packet)
    man={p.name:sha(p) for p in (weights,HERE/"head.py",HERE/"build.py",HERE/"train.py",HERE/"lots.py",HERE/"data.py",HERE/"protocol.json",target)}
    (HERE/"release.json").write_text(json.dumps(dict(files_sha256=man,bytes=len(packet),api="forecast(observation, history, worker_uid, planned_action=None, friendly_orders=None)",native_agent_replaced=False),indent=2)+"\n")
if __name__=="__main__":build()

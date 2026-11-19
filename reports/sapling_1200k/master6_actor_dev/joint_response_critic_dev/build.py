"""Attach independent, genuine Sapling-team decision weights to the byte-frozen parent."""
import base64
import json
import zlib
from data import D,PARENT,FastCarbon,sha
import hashlib
def main():
    base=PARENT.read_text();selector=base[base.rindex('\ndef _mixed_actions('):]
    assert selector.count('return baseline')==3 and selector.count('return candidate')==1
    selector=selector.replace('return baseline','return _jrc_attach(x,aux,enc,baseline)').replace('return candidate','return _jrc_attach(x,aux,enc,candidate)')
    keys='treeLifespan plantCost playerTreeProtectiveNumber plantCostInflationRatio plantCostInflationBase seizeCost workerLimit recCollectorCost recCollectorIncreaseCost recPlanterCost recPlanterIncreaseCost'.split()
    cfg=FastCarbon().config
    packed=base64.b85encode((D/'weights.npz').read_bytes()).decode()
    src=base+'\nimport io\n'+(D/'actions.py').read_text()+selector+'\n_JRC_CFG = '+repr({k:cfg[k] for k in keys})+'\n_JRC_WEIGHTS=dict(np.load(io.BytesIO(base64.b85decode('+repr(packed)+')),allow_pickle=False))\n'+(D/'serve.py').read_text()
    compile(src,'joint-response-sapling','exec');path=D/'candidate.py';path.write_text(src)
    record=dict(candidate_sha256=sha(path),parent_sha256=sha(PARENT),weights_sha256=sha(D/'weights.npz'),bytes=path.stat().st_size,
        protocol_sha256=sha(D/'protocol.json'),actions_sha256=sha(D/'actions.py'),serve_sha256=sha(D/'serve.py'))
    (D/'package.json').write_text(json.dumps(record,indent=2)+'\n');print(record)


if __name__=='__main__':main()

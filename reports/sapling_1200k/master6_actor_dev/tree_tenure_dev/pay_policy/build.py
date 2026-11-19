"""Produce the same deterministic Sapling file from the frozen base, without repacking model weights."""
import ast
import hashlib
import json
import sys
from pathlib import Path

D=Path(__file__).resolve().parent
ROOT=D.parents[3].parent
PARENT=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'


def make():
    src=PARENT.read_text();p=json.loads((D/'protocol.json').read_text())
    assert hashlib.sha256(src.encode()).hexdigest()==p['parent_sha256']
    select=src[src.rindex('\ndef _mixed_actions('):]
    assert select.count('return baseline')==3 and select.count('return candidate')==1
    select=select.replace('return baseline','return _pay_select(x,aux,enc,baseline,False)').replace('return candidate','return _pay_select(x,aux,enc,candidate,True)')
    decoder=src[src.index('def _return_decode('):src.index('\n    return labels',src.index('def _return_decode('))+len('\n    return labels')]
    decoder=decoder.replace('_return_decode(x, aux, encoded, apply_residual)','_pay_forced(x, aux, encoded, apply_residual, target_slot, target_action)')
    a=decoder.index('        if not apply_residual:');b=decoder.index('        labels[slot] = action')
    decoder=decoder[:a]+'''        if slot==target_slot:
            if slot>=10 or x['tok_kind'][slot]!=1 or not legal[target_action]:
                return None
            action=target_action
'''+decoder[b:]
    sys.path.insert(0,str(ROOT/'code'));from fastsim import FastCarbon
    cfg=FastCarbon({}).config
    keys='treeLifespan plantCost playerTreeProtectiveNumber plantCostInflationRatio plantCostInflationBase seizeCost workerLimit recCollectorCost recCollectorIncreaseCost recPlanterCost recPlanterIncreaseCost'.split()
    txt=src+'\n_PAY_CFG = '+repr({k:cfg[k] for k in keys})+'\n'+decoder+'\n'+select+'\n'+(D/'overlay.py').read_text()
    ast.parse(txt);(D/'candidate.py').write_text(txt)
    receipt=dict(candidate_sha256=hashlib.sha256(txt.encode()).hexdigest(),parent_sha256=p['parent_sha256'],protocol_sha256=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),bytes=len(txt.encode()))
    (D/'package.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt)


if __name__=='__main__':
    make()

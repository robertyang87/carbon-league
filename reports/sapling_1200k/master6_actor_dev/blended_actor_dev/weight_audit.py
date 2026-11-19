"""Independently compare trained tensors with their exact warm start."""
import hashlib
import json
from pathlib import Path

import numpy as np
HERE=Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def audit():
    protocol=json.loads((HERE/'protocol.json').read_bytes())
    parent=HERE.parent.parents[2]/'data/weights/sapling_1200k/master6_actor_v1.npz'
    assert sha(parent)==protocol['parent_weights_sha256']
    report={'warm_start_sha256':sha(parent),'arms':{}}
    with np.load(parent,allow_pickle=False) as p:
        for arm in ('candidate','control'):
            output=HERE/(arm+'.npz')
            changed={}
            frozen_count=0
            with np.load(output,allow_pickle=False) as z:
                assert set(z.files)==set(p.files)
                for key in p.files:
                    if key=='meta':
                        continue
                    old,new=p[key],z[key]
                    assert np.isfinite(new).all() and old.shape==new.shape
                    if key.split('.')[0] in protocol['train_modules']:
                        diff=new.astype('float64')-old
                        changed[key]={'count':new.size,'max_abs_shift':float(abs(diff).max()),'rms_shift':float(np.sqrt(np.mean(diff**2)))}
                    else:
                        if not np.array_equal(old,new):
                            raise ValueError(f'{arm} frozen value changed: {key}')
                        frozen_count+=old.size
            assert all(x['max_abs_shift']>0 for x in changed.values())
            report['arms'][arm]={'sha256':sha(output),'trainable_tensors':changed,'unchanged_frozen_elements':frozen_count}
    (HERE/'weight_audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    audit()

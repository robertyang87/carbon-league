"""Train again from the signed source tapes into a separate temporary destination."""
import tempfile
from pathlib import Path
import numpy as np
import train

def main():
    target=train.HERE
    records, barred=train.identity()
    manifest=train.load(target/'sources.json')
    assert records==manifest['records'] and barred==manifest['excluded']
    with tempfile.TemporaryDirectory() as tmp:
        train.HERE=Path(tmp)
        rep=train.fit(records,dict(np.load(train.BRIDGE/'rollout.npz',allow_pickle=False)))
        assert (Path(tmp)/'opponent.npz').read_bytes()==(target/'opponent.npz').read_bytes()
    assert rep==train.load(target/'fit.json')
    train.bridge.save(target/'reproduction.json',{'second_fit_byte_identical':True,'fit_record_and_training_loss_equal':True,
                      'weights_sha256':rep['weights_sha256'],'source_manifest_sha256':train.sha(target/'sources.json'),
                      'training_script_sha256':train.sha(target/'train.py'),'check_script_sha256':train.sha(Path(__file__))})
if __name__=='__main__':
    main()

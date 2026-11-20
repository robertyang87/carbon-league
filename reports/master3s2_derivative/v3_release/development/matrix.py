"""Eight complete games against original Master3s2, comparing v3 to its actual ladder v2 parent."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[1]
TEMPLATE=FAMILY/'calibration_improvement_dev/matrix.py'
spec=importlib.util.spec_from_file_location('frozen_head_to_head',TEMPLATE)
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
common=adapter.common
SEEDS=(621758606,621758607)
WORKFLOW='master3s2-v3-test-dev.yml'
CHALLENGER='016bc047554dde98844ddf4b7e8be03f52fcbe474017a0c5732d83f97485e31d'
PARENT='8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def contract():
    result=adapter.contract()
    p=common.paired
    files=result['files']
    manifest=p.read(HERE.parent/'manifest.json')
    required={'challenger':HERE.parent/'submission.py','parent':FAMILY/'ladder/prediction_feedback/package/candidate.py',
              'source_0':HERE.parent/'build.py','source_1':HERE.parent/'runtime.py','source_2':HERE.parent/'manifest.json'}
    p.require(all(files[k]==p.runner.record(v) for k,v in required.items()),'Unexpected input at a registered path')
    p.require(manifest['candidate_sha256']==CHALLENGER and manifest['parent_sha256']==PARENT
              and manifest['builder_sha256']==files['source_0']['sha256']
              and manifest['runtime_sha256']==files['source_1']['sha256']
              and manifest['base_file']==files['parent']['path'] and manifest['bytes']==files['challenger']['bytes'],
              'Submission manifest differs from the actual pair')
    files['paired_contract_template']=p.runner.record(TEMPLATE)
    result['schema']='ladder-v3-v2-original-development/1'
    result['scope']='Frozen v3 and exact online v2 each face the unmodified original on both seats of two independent development boards. Both arms use the same host per map-seat with alternating order and independent natural sampling. All eight attempts count; no formal maps or upload to ladder.'
    return result


if __name__=='__main__':
    adapter.HERE=HERE
    adapter.SEEDS,adapter.WORKFLOW,adapter.ARM_SHA,adapter.PARENT_SHA=SEEDS,WORKFLOW,CHALLENGER,PARENT
    common.HERE,common.PLAN,common.RELEASE=HERE,HERE/'protocol.json',HERE/'release.json'
    common.SEEDS,common.WORKFLOW,common.contract=SEEDS,WORKFLOW,contract
    raise SystemExit(common.main())

"""Recover an interrupted, already-admitted arena using its exact signed binaries and source files."""
import os
import sys
from shared import HERE,ROOT,read,require,sha


def verify():
    binding=read(HERE/'games/binding.json')
    current={rel:sha(ROOT/rel) for rel in binding['checked_driver_sha256']}
    require(current==binding['checked_driver_sha256'],'original signed driver changed')
    gate=read(HERE/'admission.json')
    require(gate['passed'] and sha(HERE/'admission.json')==binding['gate_sha256'],'admission changed')
    require(sha(HERE/'clearance_tip.json')==binding['clearance_tip_sha256'] and sha(HERE/'clearance_fault.json')==binding['fault_exposure_sha256'],'exposure receipt changed')
    return current


if __name__=='__main__':
    verify()
    if '--verify-only' not in sys.argv:
        os.execv(sys.executable,[sys.executable,'-u',str(HERE/'native.py')])

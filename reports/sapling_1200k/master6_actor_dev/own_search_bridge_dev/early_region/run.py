"""Run the frozen selection or independently validate its completed command tapes."""
import importlib.util
import runpy
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    if sys.argv[1:]==['verify']:
        import registered
        spec=importlib.util.spec_from_file_location('original_third_route_audit',HERE.parent/'third_route/audit.py')
        audit=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        audit.HERE=HERE
        audit.verify()
    else:
        runpy.run_path(str(HERE/'registered.py'),run_name='__main__')

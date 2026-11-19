from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'midgame_lane_dev'))
import path_audit
import importlib.util
spec=importlib.util.spec_from_file_location("lane_verify", HERE.parent/"midgame_lane_dev"/"verify.py")
verify=importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
if __name__=='__main__':
    verify.HERE=HERE
    verify.report()
    path_audit.run(HERE)

"""Run the prior independent development-map audit on this declaration."""
from pathlib import Path
import importlib.util

here = Path(__file__).resolve().parent
audit = here.parent / "tree_response_training/planter_survival/native_dev/clearance.py"
spec = importlib.util.spec_from_file_location("survival_map_audit", audit)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
source.HERE = here
if __name__ == "__main__":
    source.issue(None)

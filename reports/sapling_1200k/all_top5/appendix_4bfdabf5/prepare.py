"""Package one frozen incremental batch without changing its baseline."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from appendix_batch import prepare


if __name__ == "__main__":
    baselines=[HERE.parent/"manifest.json", HERE.parent/"appendix_499886f1/manifest.json"]
    prepare("4bfdabf588f631e74077d21fe97e267c2521e533", baselines, HERE)

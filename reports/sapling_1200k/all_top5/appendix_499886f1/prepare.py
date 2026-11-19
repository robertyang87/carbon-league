"""Package one frozen incremental batch without changing its baseline."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from appendix_batch import prepare


if __name__ == "__main__":
    prepare("499886f1a673f71a8fc3454ce9fee998d65af0c9", [HERE.parent/"manifest.json"], HERE, expected=(8,1))

"""Freeze the ten newly indexed games relative to their three registered predecessors."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from validation_delta import run

REVISION = "555c0c566bd2572032568daeb8fa79f73a4c36e7"
PREVIOUS = "4bfdabf588f631e74077d21fe97e267c2521e533"
PARENTS = (
    (".","c70e0f7953ac7b4adc04e1a768b9a72c69f6057d6eca8f5f8cb58ffb9072f486"),
    ("appendix_499886f1","707d8c302ccb4ff7f8eb9106d503a6cfaa0208c5a42b3ae02f4ac5cdd5606e4e"),
    ("appendix_4bfdabf5","a91e9aedf5ea1597edfb49b35e50d0c2f7d7c7c93cfc6fd8655fcc67fc59e58a"),
)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("/workspace/sapling-all-top5-20261119"))
    run(parser.parse_args().cache,HERE,REVISION,PREVIOUS,PARENTS)

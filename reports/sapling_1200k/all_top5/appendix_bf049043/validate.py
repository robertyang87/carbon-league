"""Verify games newly indexed by the last 11:56 snapshot against four earlier frozen ledgers."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from validation_delta import run

REVISION = "bf049043a995807e723f5345c39469aaf59050de"
PREVIOUS = "555c0c566bd2572032568daeb8fa79f73a4c36e7"
PARENTS = (
    (".","c70e0f7953ac7b4adc04e1a768b9a72c69f6057d6eca8f5f8cb58ffb9072f486"),
    ("appendix_499886f1","707d8c302ccb4ff7f8eb9106d503a6cfaa0208c5a42b3ae02f4ac5cdd5606e4e"),
    ("appendix_4bfdabf5","a91e9aedf5ea1597edfb49b35e50d0c2f7d7c7c93cfc6fd8655fcc67fc59e58a"),
    ("appendix_555c0c56","a4bd7b2cf6998a13b292f40f10e25361ba832604bec53147909c781f7c91ee2f"),
)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("/workspace/sapling-all-top5-20261119"))
    run(parser.parse_args().cache,HERE,REVISION,PREVIOUS,PARENTS)

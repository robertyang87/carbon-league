"""Verify the newly frozen strategy-score indexes relative to the previously studied 1,373 games."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from validation_delta import run

REVISION = "5484590b3fae60b3c906c6a0be9f7840c241388a"
PREVIOUS = "bd3c0021b0992e900236df9dbdf5cd05774ee11e"
PARENTS = (
    (".","c70e0f7953ac7b4adc04e1a768b9a72c69f6057d6eca8f5f8cb58ffb9072f486"),
    ("appendix_499886f1","707d8c302ccb4ff7f8eb9106d503a6cfaa0208c5a42b3ae02f4ac5cdd5606e4e"),
    ("appendix_4bfdabf5","a91e9aedf5ea1597edfb49b35e50d0c2f7d7c7c93cfc6fd8655fcc67fc59e58a"),
    ("appendix_555c0c56","a4bd7b2cf6998a13b292f40f10e25361ba832604bec53147909c781f7c91ee2f"),
    ("appendix_bf049043","bba356c4bfd2af6e2c2a038d80213e509e5a22feb63e32b71eafb7d29101356f"),
    ("appendix_bd3c0021","cdc468c2fc14dde0b402d050749d4de08b60af1a0b3794d6be9538ebdf8139b3"),
)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("/workspace/sapling-all-top5-20261119"))
    run(parser.parse_args().cache,HERE,REVISION,PREVIOUS,PARENTS)

from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'planter_terminal_choice_dev'))
import ladder_delta as ladder
A,B='a5867c07b','f1c081756'
path='data/master5s_matches.json'
if __name__=='__main__':
    previous,latest=(ladder.cards(rev,path) for rev in (A,B))
    keys=sorted(latest.keys()-previous.keys())
    with ThreadPoolExecutor(2) as pool:
        games=list(pool.map(ladder.recover,(latest[k] for k in keys)))
    (HERE/'master5s_delta.json').write_bytes(ladder.fetch.canonical(dict(start=A,end=B,indices=[path],games=games)))
    print(len(games))

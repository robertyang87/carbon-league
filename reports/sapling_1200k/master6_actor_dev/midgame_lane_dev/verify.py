"""Recompute official endpoints, team cash flows and the named carrier's response on each arm."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'two_turn_intercept_dev'))
import ledger
def report():
    rows= [ledger.check(HERE/f'seat{s}',s,a) for s in (0,1) for a in ('parent','approach')]
    ledger.sr.retain(HERE/'ledger.json',rows)
    for r in rows:
        s=r['seat'];f=r['windows']['10']['flows']
        print(s,r['arm'],r['margin'],r['deposits'][:4],[(x.get('own_delivery',0),x.get('tree_absorption',0)) for x in f],r['calls'],r['cutoffs'])
if __name__=='__main__':report()

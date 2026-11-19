from response import sr,HERE
import sys
sys.path.insert(0,str(HERE.parent/'two_turn_intercept_dev'))
from ledger import check
def run():
    rows=[check(HERE/'seat1',1,arm) for arm in ('parent','approach')]
    sr.retain(HERE/'verified.json',rows)
    for r in rows:
        print(r['arm'],r['score'],r['margin'],r['first_five'][:2],r['collisions'][:3],r['deposits'][:3],r['calls'])
if __name__=='__main__':run()

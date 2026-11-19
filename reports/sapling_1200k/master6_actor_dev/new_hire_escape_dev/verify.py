from root_check import HERE,sr
import sys
sys.path.insert(0,str(HERE.parent/'two_turn_intercept_dev'))
import ledger
def run():
    rows=[ledger.check(HERE/'seat0',0,a) for a in ('parent','approach')]
    sr.retain(HERE/'ledger.json',rows)
    for r in rows:print(r['arm'],r['margin'],r['deposits'][:6],r['calls'],r['cutoffs'])
if __name__=='__main__':run()

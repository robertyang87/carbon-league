"""Trace the largest ten-step increase in the examiner's cash lead in each original match."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'freight_toll_route_dev'))
from turns import check,sr

def run():
    seed=sr.load(HERE/'protocol.json')['seeds'][0]
    allrows=[]
    for side in (0,1):
        for arm in ('parent','candidate'):
            path=HERE/f'games/{arm}/raw_{seed}_{side}.json.gz'
            allrows.append(check(path,side))
    sr.retain(HERE/'turning_points.json.gz',allrows)
    for r in allrows:
        print(r['arm'],r['seat'],r['start'],r['end'],r['lead'],
              sorted(r['components'].items(),key=lambda p:-abs(p[1]))[:2])
if __name__=='__main__':run()

"""Summarize the examiner's external time, CPU, cutoff and depth from every native call."""
from audit import HERE,sr
from collections import Counter
from statistics import mean
def check():
    report=[]
    for r in sr.load(HERE/'scores.json'):
        path=HERE/'games'/r['arm']/f"wall_{r['seed']}_{r['seat']}.json.gz"
        doc=sr.load(path);calls=doc['calls'];depth=Counter(x['max_depth'] for x in calls)
        assert list(range(r['steps']))==[x['step'] for x in calls]
        host=doc['segments'][0]['host']
        report.append(dict(seat=r['seat'],arm=r['arm'],n=len(calls),max_depth=dict(depth),budget=doc['summary']['native_budget_seconds'],
            mean_candidates=mean(x['candidates'] for x in calls),cutoffs=sum(x['native_aborted'] for x in calls),
            mean_cpu=mean(x['process_cpu_seconds'] for x in calls),mean_wall=mean(x['public_wall_seconds'] for x in calls),
            host=host,elapsed=sr.load(path.with_name(path.name.replace('wall_','row_').replace('.json.gz','.json')))['wall_seconds']))
    sr.live.save(HERE/'timing.json',report)
if __name__=='__main__':check()

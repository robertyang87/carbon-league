from statistics import mean
from run import HERE,sr

def analyze(rows):
    all_scores=[list(r['scores'].values()) for r in rows]
    return dict(n=len(rows), parent_wins=sum(r['scores']['parent']>0 for r in rows),
                at_least_one_win=sum(max(v)>0 for v in all_scores),
                both_win=sum(min(v)>0 for v in all_scores), both_lose=sum(max(v)<0 for v in all_scores),
                hindsight_mean_best_margin=mean(max(v) for v in all_scores))

def audit():
    cases=sr.load(HERE.parent/'outcome_audit.json')['roots'];rows=[]
    for case in cases:
        scores={}
        for arm,signature in case['arms'].items():
            path=sr.ROOT/case['case']/f'live_{arm}.json'
            assert sr.sha(path)==signature['receipt_sha256']
            receipt=sr.load(path)
            assert sr.sha(path.parent/receipt['action_tape'])==receipt['action_sha256']
            scores[arm]=receipt['margin']
        rows.append(dict(case=case['case'],batch=case['batch'],scores=scores))
    parts={name:analyze(group) for name,group in [('all',rows),('previous_hold',[r for r in rows if r['batch']>6])]}
    sr.retain(HERE/'opportunity.json',dict(summary=parts,rows=rows))
    print(parts)

def verify():
    from branches import begin
    results=[]
    for entry in sr.load(HERE/'selection.json'):
        seat=entry['seat'];out=HERE/f'seat{seat}'
        q=sr.load(out/'live_protocol.json')
        raw=sr.load(sr.ROOT/q['record']['source'])
        modes={}
        for arm in ('parent','third'):
            rec=sr.load(out/f'live_{arm}.json')
            tape=out/rec['action_tape'];assert sr.sha(tape)==rec['action_sha256']
            assert rec['protocol_sha256']==sr.sha(out/'live_protocol.json') and rec['shared_init_sha256']==sr.sha(out/'live_shared_init.json')
            tail=sr.load(tape);env=begin(raw)
            for joint in raw['actions'][:entry['step']]+tail:env.step(joint)
            scores=[s.reward for s in env.states]
            assert env.done and scores==rec['score'] and rec['margin']==scores[seat]-scores[1-seat]
            assert rec['search_calls']==len(tail)-1 and tail[0][1-seat]==raw['actions'][entry['step']][1-seat]
            modes[arm]=dict(margin=rec['margin'],scores=scores,steps_after_root=rec['search_calls'],
                            searched=rec['search_completed'],aborted=rec['search_aborted'])
        results.append(dict(**entry,arms=modes,delta=modes['third']['margin']-modes['parent']['margin']))
    sr.retain(HERE/'response.json',results);print(results)

if __name__=='__main__':
    import sys
    audit()
    if '--verify' in sys.argv:verify()

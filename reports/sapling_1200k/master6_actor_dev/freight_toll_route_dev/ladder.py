"""Read every complete ranked replay ledger and compare actual freight taxes, bank and tree payments."""
from collections import defaultdict
import gzip,json
from measure import HERE,sr

def collect(include_stages=False):
    root=HERE.parents[1]/'all_top5'
    manifest=sr.load(root/'manifest.json')
    elite={p['name'] for p in manifest['top5']}
    rows=[];seen=set()
    def add(game,record,digest):
        assert game['id'] not in seen and game['sha256']==digest
        assert record['rewards']==game['score_by_seat']
        seen.add(game['id'])
        cash=record['whole']['seats']
        for s,name in enumerate(game['seat_names']):
            if name not in elite:continue
            other=1-s;b=cash[s];opp=cash[other]
            rows.append(dict(id=game['id'],seed=game['seed'],seat=s,player=name,opponent=game['seat_names'][other],
                        outcome=int(game['score_by_seat'][s]>game['score_by_seat'][other])-int(game['score_by_seat'][s]<game['score_by_seat'][other]),
                        bank=b['bank'],opponent_bank=opp['bank'],received=b['tree_from_enemy_carried'],
                        paid=opp['tree_from_enemy_carried'],tree=b['tree'],other_tree=opp['tree']))
            if include_stages:
                stages={}
                for stage in record['stages']:
                    lo=stage['a']
                    if lo not in (0,100,200):raise ValueError(f'unexpected stage {lo}')
                    stats=stage['seats']
                    stages[lo]=[stats[idx]['bank'] for idx in (s,other)]+[stats[idx]['bank_count'] for idx in (s,other)]
                rows[-1]['stages']=stages
    base={g['id']:g for g in manifest['games']}
    for path in sorted((root/'economy_data').glob('*.jsonl.gz')):
        for line in gzip.open(path):
            row=json.loads(line);add(base[row['id']],row['record'],row['replay_sha256'])
    for manifest_path in sorted(root.glob('appendix_*/manifest.json')):
        index={g['id']:g for g in sr.load(manifest_path)['games']}
        for r in sr.load(manifest_path.parent/'cases.json.gz')['rows']:
            add(index[r['id']],r['economy'],r['source_sha256'])
    assert len(seen)==1590
    return rows

def run():
    rows=collect();out=[]
    for name in sorted({r['player'] for r in rows}):
        for outcome in ('all',-1,0,1):
            group=[r for r in rows if r['player']==name and (outcome=='all' or r['outcome']==outcome)]
            if not group:continue
            out.append(dict(name=name,outcome=outcome,n=len(group),
                **{k:sum(r[k] for r in group)/len(group) for k in ('bank','opponent_bank','received','paid','tree','other_tree')}))
    names=["claude-AlphaCarbon-Master6","claude-AlphaCarbon-Master5s"]
    pools=defaultdict(list)
    for r in rows:
        if r["player"] in names and r["opponent"] not in names:pools[(r["opponent"],r["seat"],r["player"])].append(r)
    pairs=[]
    for opponent,seat in sorted({(k[0],k[1]) for k in pools}):
        arms=[pools[(opponent,seat,name)] for name in names]
        if min(map(len,arms))<2:continue
        def avg(arm,a,b):return sum(r[a]-r[b] for r in arm)/len(arm)
        pairs.append(dict(opponent=opponent,seat=seat,n=list(map(len,arms)),
                    bank_margin=[avg(arm,'bank','opponent_bank') for arm in arms],
                    tax_margin=[avg(arm,'received','paid') for arm in arms]))
    matched={}
    for what in ('bank_margin','tax_margin'):
        by=defaultdict(list)
        for row in pairs:by[row['opponent']].append(row[what][0]-row[what][1])
        matched[what]=sum(sum(v)/len(v) for v in by.values())/len(by)
    print('matched',len(by),len(pairs),matched)
    sr.retain(HERE/'matched_top_two.json',dict(strata=pairs,matched_differences=matched,players=names))
    sr.retain(HERE/'ladder_tolls.json.gz',rows)
    sr.retain(HERE/'ladder_summary.json',out)
    print([(r['name'],r['n'],round(r['received']),round(r['paid']),round(r['bank']-r['opponent_bank'])) for r in out if r['outcome']=='all'])
if __name__=='__main__':run()

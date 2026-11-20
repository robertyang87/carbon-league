"""Enumerate all historic labeled pairs and hash-derived response draws before running the teacher."""
import gzip,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
OLD=D/'action_rank_data'
CLOSE=D/'return_response_learning/close_games'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(path):return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())
def key(card):
    return f"{card['group']}_{card['seed']}_{card['seat']}_{card['step']}"
def register():
    cards=[]
    for t in read(OLD/'protocol.json')['tickets']:
        g=t['split'];stem=f"{t['seed']}_{t['seat']}_{t['step']}"
        p=OLD/g/(stem+'_root.json.gz');root=read(p)
        cards.append(dict(seed=t['seed'],seat=t['seat'],step=t['step'],terrain=t['terrain'],group=g,
                          source_sha=t['source_sha256'],root_sha=sha(p.read_bytes()),
                          root_path=str(p.relative_to(D)),status=root['status'],
                          parent=sha((D/'cold_start/candidate.py').read_bytes()),native_origin='7cf5 native H8 shortlist vs full original'))
    menu=CLOSE/'menus.json'
    for n,x in enumerate(read(menu)['rows']):
        t=x['ticket']
        cards.append(dict(seed=t['seed'],seat=t['seat'],step=t['step'],terrain=t['terrain'],group='close',
                          source_sha=t['sha256'],root_sha=sha(menu.read_bytes()),root_path=str(menu.relative_to(D)),
                          close_ordinal=n,status=x['status'],parent=sha((D/'ladder/prediction_feedback/package/candidate.py').read_bytes()),
                          native_origin='8b4c original-search finalists, close late-game filtered cohort'))
    for row in cards:
        token='/'.join(str(row[k]) for k in ('group','source_sha','root_sha','seed','seat','step'))
        row['draw_seeds']=[int(sha(f'paired-terminal-teacher-v1/{token}/{i}'.encode())[:8],16)%(2**31) for i in range(8)]
        assert len(set(row['draw_seeds']))==8
        row['native_draw_seed']=(1000003*(row['step']+1)+17*row['seat'])%(2**31)
    assert len(cards)==len(set(r['terrain'] for r in cards))==16
    doc=dict(tickets=cards,draw_count=8,discount=1,primary='Use held+close (8 maps) as two retrospective evaluation cohorts and report each, pooled and training separately. Original completed shortlist only; no refitting, no other arms. Native first/second commands frozen before actual historic two arms. 7 train, 4 previously exposed held, 4 close complete; retain the eighth originally non-H8 training ticket as skip.',
             simulation='Seat-visible history through t only; freeze two archived team orders. For each fresh SHA-derived scenario draw one root opponent order from the frozen own-belief model of the opponent, shared by A/B. Thereafter sample both native policy continuations, shared per-time random seed within A/B, until H8 or exact official terminal. Evaluate with corresponding frozen parent critic. Search timing and pruning removed only for offline 2x8 evaluation. No new matches, no future real enemy actions. Reproduce original dry seed and match each archived native H8 critic within absolute 2e-4. Failure is reported and excluded, never replaced. Original dry seed is not a member of fresh eight.',
             teacher='Average H8 over 8 independent model response seeds; tie retains native winner. Native original soft-score comparator uses archived original single H8. Also report fresh single 0 as separate sensitivity: archived two are preselected on original search, hence regression to mean. Action differences and real judge are independent of future sampled trajectories; historic outcomes were published before design. No sample size, temperature or thresholds selected by outcome.',
             outcomes='Same cost-to-go consumed for genuine joint commands. Primary pooled average official margin gain and accuracy of actual higher margin, WDL change; secondary 12-action-window actual team net bank receipts own minus foe. Report by root, cohort and map-cluster bootstrap with 10000 replicates, seed 712908. Per-pair mean, paired differences sample SE across 8 scenario seeds and 95% t7 intervals conditional on the model only; observed natural labels have one response per arm, no true per-root SE. Pre-specified positive pilot only if pooled raw mean delta>0, positive point gain in both held and close and strictly more correct than native across pooled; uncertainty governs any scaling inference.')
    (HERE/'protocol.json').write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(sha((HERE/'protocol.json').read_bytes()))
if __name__=='__main__':register()

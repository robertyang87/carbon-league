from pathlib import Path
import sys,json,gzip,hashlib
HERE=Path(__file__).resolve().parent;GROUP=HERE.parent;ACTOR=GROUP.parent;ROOT=ACTOR.parents[2]
sys.path.insert(0,str(GROUP));import bridge
PARENT=ACTOR/'deployed_mix_policy_gradient_dev/full_episode/selected.py'
if __name__=='__main__':
    roots=bridge.load(GROUP/'root_scores.json.gz');reg={}
    for r in roots:
        v=bridge.load(ROOT/r['case']/'live_protocol.json');c=v['record'];reg[c['source']]=dict(path=c['source'],sha=c['source_sha256'],seed=c['seed'],seat=c['seat'],batch=r['batch'],terrain=r['physical_terrain'],split='held' if r['batch']>=7 else 'fit')
    assert len(reg)==32 and len({c['terrain'] for c in reg.values()})==16
    bridge.save(HERE/'protocol.json',dict(sources=list(reg.values()),parent=bridge.sha(PARENT),data='Both public seat histories from 32 complete actual matches, labels actual issued legal joint-action prefixes and terminal official margin/utility. Train first six two-map batches. Hold last two batches (4 separate terrains, 8 games) for outcome AND archived real single-planter intervention ranking. No native matchup, no 342/343, no closed weights. Sample decision indices 0,8,...,296. Single fit, fixed final epoch, no held choice.',
       network='Shared local+4-neighbour board mean -> 16 channels tanh, average and max global pooling + 20 scalars -> 16. Bias-free action MLP uses local destination, role, action, sequential prefix features and context. Dual value heads learn official end U and residual (terminal margin-current cash margin)/4000. Score=0.5*(estimated U centered) + 0.5*tanh((current margin+4000*residual)/2000); use exact utility mapping on terminal. One NumPy pack; paired H1/H3/H5 full rounds with 4->3->2 (pinned parent), retain last complete best, soft internal timeout .68, hard .80. Root alternatives share common enemy root and modeled later responses independent per branch. Root starts from frozen true ambient 09b4 blend, one likely single-slot fork and joint pair of nearby/contested workers with real re-decoding/guard. No privilege or examiner inference at runtime.',
       gate='Compare actual isolated 16 held single-planter terminal signs vs frozen estimator; 8-step teacher and all-map results are not gate tuning. Follow held normal entry, measure multi-unit changes, public leakage test, fallback, clock; any absence of these or unfavorable held terminal result withhold recommendation. All diagnoses cost CPU2 only.'))

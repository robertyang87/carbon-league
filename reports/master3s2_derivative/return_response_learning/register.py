"""Freeze source terrain partition without loading any trajectory or outcome."""
from pathlib import Path
import hashlib, json
here=Path(__file__).resolve().parent
parent=here.parent
arc=parent/'opponent_policy/protocol.json'; pairs=parent/'action_rank_data/protocol.json'
book=json.loads(arc.read_bytes()); expr=json.loads(pairs.read_bytes()); paired={t['terrain'] for t in expr['tickets']}
# Ordinary validation consists of all four historical held maps. The pair study drew four maps from there.
# Keeping it physically disjoint from every supervised episode is necessary, even when the pair reveals only one root.
source=[g for g in book['games'] if g['split']=='val' or g['terrain'] not in paired]
assert not ({g['terrain'] for g in source if g['split']=='train'} & (paired|{g['terrain'] for g in source if g['split']=='val'}))
assert {g['seed'] for g in source}.isdisjoint({621758608,621758609})
obj=dict(ordinary=source,pairs=expr['tickets'],source_sha=hashlib.sha256(arc.read_bytes()).hexdigest(),pair_source_sha=hashlib.sha256(pairs.read_bytes()).hexdigest(),
 study='Action-conditioned prediction of 12-move own-carrier receipt in original expert trajectories. Existing paired-branch labels include 8 registered train / 4 previously exposed held maps; 1 train root historically not H8. No new maps or outcomes in this study.',
 ordinary_target='Each originally operated expert collector with positive load at t=24,27,..,294 incl any >0, label=min(1, first positive actual own_delivery for same UID in next min(12,299-t) steps/current cargo). No receipt=0. Arrival alive or a transfer to another UID alone gives no credit. Same-worker credit is truncated to current cargo. Official cash path verified on every map.',
 features='Worker moves physically from five commands predicted within own joint commands; distance, drift, edge shielding, friendly planned collision, own base spawn, nearby lighter enemies, cargo, remaining and room, enemy-base hazard. No opponent actual action or future observation.',
 fitter='8 hidden tanh, sigmoid output, seeded-37011 L-BFGS squared proportion loss with mean map weighting; L2=.002 on matrices, max 70 iterations. Matched state-only identical network, and fixed linear distance-to-deposit, as references. No action-effect causal interpretation of logged fit.',
 pairs_contract='From H8 point differences of top original two joint orders, add beta*delta predicted own receipt/3000. beta via nonnegative ridge regression (prior zero; mean d^2 loss+0.0001*beta^2, cap3). Target paired _terminal(raw margin) difference minus frozen native delta. Evaluate raw official margin, own and net deposits next12, and WDL. Run both-arm observed responses. No fitting on held. Original held outcomes were previously published for other heads; this is retrospective. No out-of-domain roots used.',
 qualification='Ordinary held weighted model Brier below fixed distance and state-only reference, plus 4 complete paired held maps, at least 3 with changed model predictions (absolute correction >1e-6), two or more genuine correct ranking flips, no new wrong flips, positive actual mean raw margin gained vs original, lower paired native-scale delta MSE, beta>0. No post-hoc refits. Per-ticket results and failure kept.',
 deployment='Only original same-depth H8 top two; endgame t279 or285, pair_scale stored in value units per actual point. Empty owned positive load gives zero. Net root score remains native; no other times/positions certified.')
(here/'protocol.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True)+'\n')

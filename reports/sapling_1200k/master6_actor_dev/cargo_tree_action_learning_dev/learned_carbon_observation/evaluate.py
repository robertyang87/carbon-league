"""Held-terrain map error and actionable local ranks on the archived physical action prefix."""
from collections import defaultdict
import numpy as np
from common import F, HERE, PROTOCOL, selected, save, sha
from runtime import _carbon_inputs, _carbon_view

params = np.load(HERE / "weights.npz")
network = [params[k] for k in "wbva"]
control = [params["c"]]
names = ["parent", "constant", "student"]
roles = ["collector", "planter"]
cards, records = [], []
n = F.DEST[:,1:]
pairs = np.triu_indices(5, 1)

def err_counters():
    return np.zeros(4, np.float64)

def rank_counters():
    return np.zeros((3, 6), np.float64)

def rank(vals, valid, counter):
    if valid.sum()<2:
        return
    v = vals[:, valid]
    l, r = np.triu_indices(v.shape[1], 1)
    d = v[:,l] - v[:,r]
    truth = np.sign(d[0])
    informative = np.abs(d[0]) > 1e-5
    for i, score in enumerate(v[1:]):
        pred = np.sign(d[i+1])
        hit = pred[informative] == truth[informative]
        tie = pred[informative] == 0
        high = score == score.max()
        counter[i] += [informative.sum(), hit.sum() + .5*tie.sum(), tie.sum(), 1, v[0].max()-v[0,high].mean(), float(np.any(np.abs(v[0,high]-v[0].max())<1e-5))]

def result(counters, rank=False):
    if rank:
        return {k:{arm:{'informative_pairs':int(a[0]),'accuracy_half_ties':float(a[1]/a[0]) if a[0] else None,'pred_tie_rate':float(a[2]/a[0]) if a[0] else None,'choices':int(a[3]),'uniform_argmax_regret_carbon':float(a[4]/a[3]) if a[3] else None,'any_top_truth_optimum':float(a[5]/a[3]) if a[3] else None} for arm,a in zip(names,c)} for k,c in counters.items()}
    return {k:dict(cells=int(v[0]),mae={arm:float(e/v[0]) if v[0] else None for arm,e in zip(names,v[1:])}) for k,v in counters.items()}

def all_counters(fn):
    return defaultdict(fn)

all_err = all_counters(err_counters)
all_rank = all_counters(rank_counters)
for card in PROTOCOL["cards"]:
    if card["split"] != "val":
        continue
    err = all_counters(err_counters)
    ranks = all_counters(rank_counters)
    board_count = 0
    for feat,x,truth,obs,commands,cfg in selected(card):
        board_count += 1
        feat_array, eligible = _carbon_inputs(x,feat,F)
        xc = _carbon_view(x,feat,F,control)
        xs = _carbon_view(x,feat,F,network)
        public = x["maps"][3].reshape(-1)>0
        assert np.all(eligible <= ~public)
        preds = np.stack([b["maps"][0].reshape(-1) for b in [x,xc,xs]]).astype(float)*(100/255)
        assert np.array_equal(preds[0,~eligible],preds[1,~eligible])
        assert np.array_equal(preds[0,~eligible],preds[2,~eligible])
        error = np.abs(preds - truth)
        local = np.zeros((2,225), bool)
        mask = x["tok_mask"][:10]
        loc = x["tok_pos"][:10][mask]
        types = x["tok_kind"][:10][mask]
        for role in (0,1):
            dest = F.DEST[loc[types==role]]
            targets = dest if role==0 else n[dest]
            local[role, targets.flatten()] = True
        criteria = {"all_hidden":~public,"eligible":eligible,"visible":public,
                    "hidden_previously_seen":~public&(x["maps"][2].reshape(-1)>0),
                    "collector_local_hidden":~public&local[0],"planter_local_hidden":~public&local[1]}
        for name, valid in criteria.items():
            v = np.array([valid.sum(),*error[:,valid].sum(axis=1)])
            err[name] += v
            all_err[name] += v
        vals = np.concatenate([truth[None,:],preds])
        prev = F.Prefix(x, feat.aux)
        labels = feat.labels(commands)
        trees = x["maps"][16:18].reshape(2,225).any(axis=0)
        bases = x["maps"][14:16].reshape(2,225).any(axis=0)
        own = obs["players"][feat.seat]
        count = sum(1 for team in obs["players"] for p,age in team[3].values() if age < cfg["treeLifespan"])
        fee = cfg["plantCost"] + (cfg["plantCostInflationRatio"]*cfg["plantCostInflationBase"]**count if len(own[3])>cfg["playerTreeProtectiveNumber"] else 0)
        for slot in np.flatnonzero(x["tok_mask"][:F.OWN]):
            if slot<10:
                _,_,legal = prev.feats(slot)
                role = x["tok_kind"][slot]
                dest = F.DEST[x["tok_pos"][slot]]
                feasible = ~trees[dest]&~bases[dest]
                if role==1:
                    feasible &= own[0] >= fee
                    values = vals[:,n[dest]].mean(axis=2)
                    hidden = ~public[n[dest]].all(axis=1)
                    active = eligible[n[dest]].any(axis=1)
                else:
                    values = vals[:,dest]
                    hidden = ~public[dest]
                    active = eligible[dest]
                valid = legal.astype(bool)&feasible&hidden
                key = roles[role] + "_hidden"
                rank(values,valid,ranks[key])
                rank(values,valid,all_rank[key])
                if np.any(active & valid):
                    key = roles[role] + "_hidden_with_eligible_alternative"
                    rank(values,valid,ranks[key])
                    rank(values,valid,all_rank[key])
                if valid.sum()>=2:
                    records.append([card["seed"],obs["step"],feat.seat,int(slot),roles[role],
                                    [int(v) for v in valid], [int(v) for v in active], values.tolist()])
            prev.advance(slot,int(labels[slot]))
    cards.append(dict(seed=card["seed"],terrain=card["terrain"],perspective_steps=board_count,errors=result(err),ranks=result(ranks,True)))
    print("held",card["seed"],cards[-1]["errors"]["eligible"],flush=True)
macro = {k:{arm:float(np.mean([c['errors'][k]['mae'][arm] for c in cards])) for arm in names} for k in all_err}
save(HERE/"heldout.json",dict(protocol_sha256=sha((HERE/"protocol.json").read_bytes()),sample='every 6th decision including step zero, both fixed-tape public perspectives',role_masks='canonical legal moves under recorded prefix; collector on any non-tree non-base destination; planter on affordable unoccupied-by-tree/base soil; planters score neighbor mean and require at least one presently hidden neighbor; collector destinations must be hidden',tie_semantics='informative pair truths differ by >1e-5 carbon, predictions considered tied when exactly equal; regret is uniform on estimated top ties',point_weighted=result(all_err),map_equal_weight_mae=macro,rank_totals=result(all_rank,True),cards=cards))
save(HERE/"rank_rows.json.gz",dict(fields=['seed','step','physical_seat','slot','role','economically_legal_hidden','has_eligible_cell','values'],board_values=['truth','parent','constant','student'],rows=records))

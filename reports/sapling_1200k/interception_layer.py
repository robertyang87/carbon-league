"""Optional Sapling collector action override from enemy move probabilities."""
import math

CARGO_THRESHOLD = 20.
MIN_GAIN = 25.
MAX_LOGIT_GAP = 3.


def choose_actions(model, features, tensors, obs, previous, cfg, weights, intent):
    seat = obs['player']
    own, foe = obs['players'][seat], obs['players'][1-seat]
    original, _, _ = model.act(tensors, features.aux, teacher=2)
    original_actions = features.to_actions(original)
    if not own[2] or not foe[2]:
        return original_actions, original_actions, []
    collectors = [(cell,cargo) for cell,cargo,kind in own[2].values() if kind == 'COLLECTOR']
    meaningful = any(kind=='COLLECTOR' and max(cargo,mass)>=CARGO_THRESHOLD and intent.DIST[cell,other]<=2
                     for cell,cargo,kind in foe[2].values() for other,mass in collectors)
    if not meaningful:
        return original_actions, original_actions, []
    probabilities = intent.predict(obs, weights, previous)
    arriving = {}
    for uid,(cell,mass,kind) in foe[2].items():
        if kind == 'COLLECTOR':
            for action,p in enumerate(probabilities[uid]):
                arriving.setdefault(int(intent.DEST[cell,action]),[]).append((mass,float(p)))
    home = set(own[1].values())
    enemy_home = set(foe[1].values())
    changes = []

    def hook(slot, logits, prefix, action):
        if slot >= 10 or changes or tensors['tok_kind'][slot] != 0:
            return None
        uid = features.own_uid[slot]
        cell, own_cargo, _ = own[2][uid]
        def terms(a):
            world = int(intent.DEST[cell,a if not seat else (0,3,4,1,2)[a]])
            threats = arriving.get(world,())
            survival = math.prod(1-p for mass,p in threats if mass <= own_cargo)
            capture = survival * sum(mass*p for mass,p in threats if mass > own_cargo)
            at_risk = (1-survival) * own_cargo
            # Losing a worker costs at least one hire. Deposits are protected;
            # the extra cash term values liquidity while avoiding double-counted cargo.
            estimate = capture - at_risk - (1-survival)*cfg['recCollectorCost']
            if world in home:
                estimate += .5 * own_cargo * survival
            if a == 0 and world not in home:
                estimate += cfg['initialCollectRate'] * obs['carbon'][world] * survival
            return world,capture,at_risk,estimate
        base = terms(action)
        best = None
        for alternative in range(5):
            if alternative == action or logits[alternative] < logits[action]-MAX_LOGIT_GAP:
                continue
            dest,capture,risk,value = terms(alternative)
            canon = features.cpos(dest)
            if dest in enemy_home or prefix.reserved[canon] or prefix.undecided[canon]:
                continue
            if max(capture-base[1], base[2]-risk) < CARGO_THRESHOLD:
                continue
            gain = value-base[3]
            if gain < MIN_GAIN or (best and gain <= best[0]):
                continue
            best = (gain,alternative,dict(uid=uid,action=int(action),alternative=int(alternative),
                                          expected_capture_delta=capture-base[1],expected_cargo_loss_avoided=base[2]-risk))
        if best:
            changes.append(best[2])
            return best[1]
        return None

    labels,_,_ = model.act(tensors, features.aux, teacher=2, hook=hook)
    chosen = features.to_actions(labels)
    if not changes:
        return original_actions,original_actions,[]
    def arrivals(actions):
        groups={}
        for uid,(pos,_,_) in own[2].items():
            dest=int(intent.DEST[pos,intent.MOVE_ID[actions.get(uid) or 'NONE']])
            groups.setdefault(dest,[]).append(uid)
        for uid,pos in own[1].items():
            if actions.get(uid) in ('RECCOLLECTOR','RECPLANTER'):
                groups.setdefault(pos,[]).append(uid)
        return {(pos,tuple(sorted(ids))) for pos,ids in groups.items() if len(ids)>1}
    if arrivals(chosen)-arrivals(original_actions):
        return original_actions,original_actions,[]
    return original_actions,chosen,changes

"""Use exactly one paired root correction after native comparisons finish, before returning an action."""
def _rr_in_play(players,hire):
    return all(p[2] or p[3] or (p[1] and p[0]>=hire) for p in players)


def _rr_select(original, alive, mean, depths, run, cands, step, me, players, depth, rolls, cut, start, clock, score, scale, epsilon):
    if step not in (279,285) or depth!=8 or rolls!=1 or cut or len(alive)<2:
        return original,"outside_native_support",None
    if clock()-start>_S_ABORT-.010:
        return original,"budget",None
    try:
        if not np.isfinite(scale) or scale<=0 or not any(w[2]=="COLLECTOR" and w[1]>0 for w in players[me][2].values()):
            return original,"unsupported_head_or_cargo",None
        if not all(depths.get(i)==depth and np.isfinite(mean[i]) for i in alive):
            return original,"unmatched_native_depth",None
        first,second=sorted(alive,key=mean.__getitem__,reverse=True)[:2]
        if first!=original or cands[first]==cands[second]:
            return original,"protected_incumbent_or_same_order",None
        boards={b[0]:b for b in run if b[1]==0 and b[0] in (first,second)}
        if len(boards)!=2 or any(len(b)<5 or not b[4] or b[2]["step"]!=step+depth for b in boards.values()):
            return original,"terminal_or_unknown_descendant",None
        pred=score(players,me,step,[cands[first],cands[second]])
        if len(pred)!=2 or not all(np.isfinite(x) for x in pred):
            return original,"unknown_prediction",None
        correction=float(scale*(pred[1]-pred[0]))
        if not np.isfinite(correction) or clock()-start>=_S_ABORT:
            return original,"late_or_nonfinite",None
        margin=float(mean[second]+correction-mean[first])
        evidence=dict(indices=[first,second],predicted_bank=list(map(float,pred)),native_gap=float(mean[first]-mean[second]),correction=correction)
        # A0's original safety margin remains in force when a new challenger would displace A0.
        threshold=epsilon if original==0 else 0.
        return (second if margin>threshold else original),"compared",evidence
    except Exception:
        return original,"missing_or_unsupported",None

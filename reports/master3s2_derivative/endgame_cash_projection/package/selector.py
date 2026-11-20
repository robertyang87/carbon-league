def _cash_pick(incumbent, alive, mean, depth_by_id, boards, orders, depth, rolls, cut, step, seat, cfg, start):
    """Re-rank only the two actual surviving native H8 orders. The unit of both scores is cash."""
    pair = None
    try:
        if cut or depth != 8 or rolls != 1 or len(alive) < 2 or 299-step not in (14,20):
            return incumbent, "not_applicable", pair, None
        if _time.perf_counter()-start > _S_ABORT-.005:
            return incumbent, "time_before", pair, None
        if cfg.get("episodeSteps",300)!=300 or cfg.get("size",15)!=15 or cfg.get("treeLifespan",50)!=50:
            return incumbent, "different_rules", pair, None
        ranked=sorted(alive,key=mean.__getitem__,reverse=True)
        if incumbent!=ranked[0] or not all(depth_by_id.get(i)==8 and _cash_math.isfinite(mean[i]) for i in alive):
            return incumbent, "native_selection", pair, None
        first,second=ranked[:2]
        pair=(first,second)
        if orders[first]==orders[second]:
            return incumbent, "same_order", pair, None
        selected={b[0]:b for b in boards if b[0] in pair and b[1]==0}
        if len(selected)!=2:
            return incumbent, "missing_scenario", pair, None
        hire=min(cfg.get("recCollectorCost",30),cfg.get("recPlanterCost",30))
        scores=[]
        for index in pair:
            board=selected[index]
            leaf=board[2]
            # Postponing the checks to the two finalists avoids scoring pruned branches.
            # An early extinction may later look solvent if the native engine played on.
            if len(board)<5 or len(board[4])!=8 or leaf["step"]!=step+8 or leaf["step"]>=299:
                return incumbent, "terminal_or_missing_history", pair, None
            if any(not p[2] and not p[3] and (not p[1] or p[0]<hire) for roster in board[4] for p in roster):
                return incumbent, "official_extinction", pair, None
            players=leaf["players"]
            x=_cash_features(players,seat,299-leaf["step"])
            score=players[seat][0]-players[1-seat][0]+_cash_residual(x,_CASH_HEAD)
            if not _cash_math.isfinite(score):
                return incumbent, "nonfinite", pair, None
            scores.append(score)
        if _time.perf_counter()-start > _S_ABORT:
            return incumbent, "time_after", pair, scores
        return (second if scores[1]>scores[0] else incumbent), "compared", pair, scores
    except Exception:
        return incumbent, "fallback_error", pair, None

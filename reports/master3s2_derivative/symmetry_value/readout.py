_SYM_DIAGNOSTIC = None


def _symmetry_values(boards, policy, alive, values, depths, net, me, t0):
    # Called before any further play(), using the exact X just scored by the primary look().
    clock = _time.perf_counter
    trace = _SYM_DIAGNOSTIC
    if trace is not None:
        trace.update(finalists=list(alive), before=[float(values[c, 0]) for c in alive],
                     stage="ineligible", evaluated=0, elapsed_total=clock()-t0)
    count = len(boards)
    if (_S_M != 1 or _S_MMAX != 1 or not 0 < count <= _S_ROWS or
            len(alive) < 2 or len(policy) != count or
            any(depths[c] != _S_D2 for c in alive)):
        return
    # The source search marks time-limit terminals. Its simulator can also contain an already
    # eliminated side before that time limit. Preserve the entire parent comparison in that case.
    rules = _RX_CFG[1]
    min_recruit = min(float(rules["recCollectorCost"]), float(rules["recPlanterCost"]))
    for b in boards:
        if any(not workers and not trees and (not centers or cash < min_recruit)
               for cash, centers, workers, trees in b[2]["players"]):
            if trace is not None:
                trace["stage"] = "elimination_on_primary_path"
            return
    # The primary pass has already measured this row shape. Count all spent time from public entry.
    predicted = 2.0 * (_S_CFIX + count * max(_S_COST[0], _S_COST[2]))
    deadline = t0 + _S_ABORT
    if clock() + predicted + 0.030 > deadline:
        if trace is not None:
            trace["stage"] = "budget_before_transform"
        return
    # Terminal boards are absent from 'boards'; their exact primary values are untouched.
    X = np.stack([item[2] for item in policy])
    rotated = rotate_features(X)
    scalars = np.stack([_scalars(b[2]['players'], b[2]['step'], me) for b in boards])
    if clock() + predicted + 0.015 > deadline:
        if trace is not None:
            trace["stage"] = "budget_after_transform"
        return
    other = net(rotated, _SID, scalars)[2]
    if trace is not None:
        trace.update(stage="evaluated", evaluated=count,
                     board_ids=[b[0] for b in boards], rotated=[float(x) for x in other])
    if other.shape != (count,) or not np.isfinite(other).all():
        if trace is not None:
            trace["stage"] = "nonfinite"
        return
    replacement = {(b[0], 0): 0.5*(values[b[0], 0]+float(x)) for b,x in zip(boards, other)}
    if clock() + 0.015 > deadline:
        if trace is not None:
            trace["stage"] = "late_discarded"
        return
    values.update(replacement)
    if trace is not None:
        trace.update(stage="committed", after=[float(values[c, 0]) for c in alive],
                     elapsed_total=clock()-t0)

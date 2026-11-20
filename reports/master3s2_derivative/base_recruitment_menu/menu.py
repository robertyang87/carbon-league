def _recruitment_alternative(players, seat, base_logits, baseline, existing):
    """Highest prior untested one-base change, preserving the worker team order."""
    workers, _, bases, cash = parse(players)
    own = [w for w in workers.values() if w[3] == seat]
    owned_bases = {bid: pos for bid, pos in bases[seat]}
    occupied = {move(w[0], WIDX.get(baseline.get(wid, 'NONE'), 0))
                for wid, w in workers.items() if w[3] == seat}
    # The current game configuration has fixed 30-carbon recruitment for either role.
    recruited = sum(bid in baseline for bid in owned_bases)
    options = []
    for bid, pos in bases[seat]:
        current = BIDX.get(baseline.get(bid, 'NONE'), 0)
        for index in range(len(BACT)):
            if index == current:
                continue
            if index:
                count = recruited + (current == 0)
                if len(own)+count > 10 or 30*count > cash[seat] or pos in occupied:
                    continue
            trial = dict(baseline)
            if index:
                trial[bid] = BACT[index]
            else:
                trial.pop(bid, None)
            key = tuple(sorted(trial.items()))
            if key not in existing:
                priority = float(base_logits[pos, index]-base_logits[pos, current])
                options.append((priority, trial))
    return max(options, key=lambda option: option[0])[1] if options else None

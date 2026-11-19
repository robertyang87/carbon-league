"""Measure official outcomes, elapsed clock, real cash and role activity across the complete naked-network exam."""
import numpy as np
import compare as c

H=c.HERE;r=c.r


def summarize(row):
    seat=row['seat'];seed=row['seed'];d=H/'games'/row['arm']
    rawfile=d/f"raw_{seed}_{seat}.json.gz";wallfile=d/f"wall_{seed}_{seat}.json.gz"
    g=r.load(rawfile);w=r.load(wallfile)
    assert r.sha(rawfile)==row['raw_sha256']==w['raw_sha256'] and r.sha(wallfile)==row['wall_sha256']
    sim=r.FastCarbon(g['game_configuration']).reset(seed=seed)
    assert list(sim.carbon)==g['initial_full_carbon']
    decisions={'collector':0,'planter':0,'base_recruit':0};standing=[]
    for pair in g['actions']:
        obs=sim.obs(seat);z=obs['players'][seat];standing.append([len(g1[3]) for g1 in obs['players']])
        for unit,a in pair[seat].items():
            if unit in z[1]:decisions['base_recruit']+=1
            else:decisions[z[2][unit][2].lower()]+=1
        sim.step(*pair)
    assert len(w['calls'])==len(g['actions'])==row['actual_opponent_calls'] and not g['errors']
    clocks=np.array([x['public_wall_seconds'] for x in w['calls']])
    cash=r.cash_summary(g,seat,0,g['actions'],list(sim.cash))
    return dict(arm=row['arm'],seat=seat,score=row['score_by_seat'],official_margin=row['margin'],steps=row['steps'],extinct=row['extinction'],result=row['result'],
        original_calls=len(clocks),examiner_public_seconds={'median':float(np.median(clocks)),'p95':float(np.quantile(clocks,.95)),'max':float(clocks.max())},
        original_native_clock=w['summary'],cash=cash,mean_visible_trees_by_seat=np.mean(standing,axis=0).tolist(),nonempty_commands_by_role=decisions,raw_sha256=r.sha(rawfile))


if __name__=='__main__':
    rows=[summarize(v) for v in r.load(H/'scores.json')];by={(v['arm'],v['seat']):v for v in rows}
    comparison={str(s):dict(margin_delta=by['candidate',s]['official_margin']-by['parent',s]['official_margin'],
        net_deposit_delta=by['candidate',s]['cash']['delivery_margin']-by['parent',s]['cash']['delivery_margin'],
        tree_income_margin_delta=by['candidate',s]['cash']['tree_income_margin']-by['parent',s]['cash']['tree_income_margin']) for s in (0,1)}
    r.write(H/'accounting.json',dict(protocol=r.sha(H/'protocol.json'),scores_sha256=r.sha(H/'scores.json'),rows=rows,same_seat=comparison))

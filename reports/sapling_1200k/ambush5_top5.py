"""Transfer the frozen public ambush probe to 16 pre-existing held-out Top5 games."""
import argparse
import json
from pathlib import Path

from ambush5_probe import H, HERE, arena, evaluate_roots, write_archive
from interception_fit import own_view, read_game


def summarize(rows):
    return dict(attempts=len(rows),
                observed_crossing=sum(r['forecast_crossing_observed'] for r in rows),
                captured=sum(r['target_captured']>0 for r in rows),
                acquired_by_parent=sum(r['parent_target_receipt']>0 for r in rows),
                missing_future_command_roots=sum(r['missing_command_ids']>0 for r in rows),
                retained=round(sum(r['target_cargo_still_carried'] for r in rows),4),
                banked=round(sum(r['target_cargo_banked'] for r in rows),4),
                delta={k:round(sum(r['delta'][k] for r in rows),4) for k in (
                    'own_cash','enemy_cash','own_cargo','enemy_cargo','own_worker_deposits','enemy_worker_deposits','enemy_cargo_acquired')})


def run(folder,out,forecaster=None):
    selected=json.loads((HERE/'interception_fit.json').read_text())['game_ids']['val']
    manifest=json.loads((HERE/'top5_h2h_data.json').read_text())['games']
    index={r['id']:r for r in manifest}
    rows=[]
    for game_id in selected:
        item=index[game_id]
        if item['split']!='val':
            raise ValueError('invalid hold-out split')
        g=read_game(folder,item)
        steps=g['steps']
        sim=arena.exam.FastCarbon(g['configuration']).load_observation(steps[0][0]['observation'])
        if sim.config['smeltCost'] or sim.config['moveCost']:
            raise ValueError('zero-cost transfer required')
        states=[]
        orders=[]
        for tick,frame in enumerate(steps):
            if tick:
                joint=[agent['action'] or {} for agent in frame]
                orders.append(joint)
                sim.step(*joint)
            truth=frame[0]['observation']
            seen=sim.full_observation()
            if (seen['players']!=truth['players'] or seen['carbon']!=truth['full_carbon'] or
                seen['trees']!=truth['trees'] or seen['step']!=truth['step']):
                raise ValueError(f'archive differs at {game_id}:{tick}')
            states.append(sim.get_state())
        roots=range(item['seed']%H,len(orders)-H+1,H)
        for side in (0,1):
            observed=[own_view(frame,side) for frame in steps]
            r=evaluate_roots(sim,states,observed,orders,side,roots,('public',),forecaster)['public']
            rows.append(dict(id=game_id,seed=item['seed'],seat=side,git_blob=item['git_blob'],sampled_roots=len(roots),model=item['models'][side],summary=summarize(r),attempts=r))
        print(game_id,rows[-2]['summary']['captured'],rows[-1]['summary']['captured'],flush=True)
    result={'scope':'On each map use its seed modulo five as starting offset; windows are disjoint in time within a seat. Seat outcomes on the same map share trajectories. Both historical agents keep recorded subsequent orders in the counterfactual. This is a development diagnostic.',
            'all_roots':sum(r['sampled_roots'] for r in rows),'games':rows,'summary':summarize([t for r in rows for t in r['attempts']])}
    write_archive(out,result)
    print(json.dumps(result['summary'],indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replays',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    run(args.replays,args.out)


if __name__=='__main__':
    main()

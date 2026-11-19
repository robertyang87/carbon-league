"""Separate access to publicly promising planting cells from the decision to spend once there."""
from collections import Counter
import gzip
import json
from select_roots import HERE,ROOT,load,retain
from turning_point_ledger import CashEvents,FastCarbon

def promising(a):
    return (a['current_site_type']=='plant' and a['site_affordable_on_present_budget'] and a['entry_legal'] and
            not a['friendly_conflict'] and a['enemy_collector_distance']>=2 and a['enemy_planter_distance']>=3 and
            a['observable_resource_sum']>=a['price'])

def build():
    source=load(HERE/'source_receipt.json')
    paid=set()
    for g in source['games']:
        raw=load(ROOT/g['raw_path'])
        sim=FastCarbon(raw['game_configuration']).reset(seed=g['seed'])
        sim.cash=CashEvents(sim.cash)
        for t,pair in enumerate(raw['actions']):
            sim.cash.events.clear()
            sim.step(*pair)
            for e in sim.cash.events:
                if e[0]=='plant' and e[1]==g['seat'] and e[2]<0:
                    paid.add((g['seed'],g['seat'],t,e[3]))
    by_map={str(s):Counter() for s in source_seeds(source)}
    with gzip.open(HERE/'opportunities.jsonl.gz','rt') as fp:
        for line in fp:
            r=json.loads(line);a=r['actions']
            stats=by_map[str(r['seed'])]
            stats['all_active_planter_roots']+=1
            moves=[m for m in a if m!='NONE' and promising(a[m])]
            here=promising(a['NONE'])
            if here:
                stats['on_promising_cell']+=1
                stats['on_promising_cell_v1_intended_pay']+=int(r['parent_command']=='NONE')
                stats['on_promising_cell_realized_payment']+=int((r['seed'],r['seat'],r['t'],r['uid']) in paid)
                stats['on_promising_cell_v1_moved']+=int(r['parent_command']!='NONE')
            elif moves:
                stats['promising_neighbor_only']+=1
                stats['chose_to_move_to_promising_neighbor']+=int(r['parent_command'] in moves)
                stats['did_not_move_to_promising_neighbor']+=int(r['parent_command'] not in moves)
            else:
                stats['neither_current_nor_adjacent_promising']+=1
    total=Counter()
    for stats in by_map.values():total.update(stats)
    if total['all_active_planter_roots']!=total['on_promising_cell']+total['promising_neighbor_only']+total['neither_current_nor_adjacent_promising']:
        raise ValueError('opportunity partition incomplete')
    doc={'definition':'Own planter, an empty site with true v1-prefix entry legal, present budget after own requests covers phase-zero planting price, no friendly conflict in proposed destination, nearest enemy collector >=2, nearest enemy planter >=3, sum of currently visible non-tree/base resource neighbors >= planting price; MOVE estimates arrival only. Cash events are actual source-tape receipts, with source map/seat/turn/worker identity. All turns are included, without terminal utility cutoff; public coverage is a heuristic and not an independent causal label.','by_map':by_map,'total':total}
    retain(HERE/'position_coverage.json',doc)
    print(dict(total))

def source_seeds(source):
    return sorted({g['seed'] for g in source['games']})

if __name__=='__main__':
    build()

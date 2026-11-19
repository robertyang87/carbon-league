"""Classify same-turn hazards in sealed pay/defer pairs. Actual enemy commands are outcome evidence only."""
import argparse
from collections import Counter
import json
from pathlib import Path

from opportunities import MOVES, _ROUTE_DEST, _ROUTE_DIST
from select_roots import ROOT, clean, load, retain, sha
from branches import begin


def analyze(area):
    rows=[]
    for r in load(area/'training_labels.json')['pairs']:
        root_path=area/r['features'];root=load(root_path);case=root_path.parent
        start=root['public_at_root'];s=r['seat'];u=r['focal_uid'];cell=r['cell']
        anchor=root['candidate']['tape_sha256']
        native={side:load(case/('live_'+r[side]['native_arm']+'.json')) for side in ('pay','defer')}
        if len({v['original_trajectory_sha256'] for v in native.values()}|{anchor})!=1 or sha(root_path)!=r['root_sha256']:
            raise ValueError('unbound intervention')
        source=ROOT/load(case/'live_protocol.json')['record']['source'];tape=load(source)
        if sha(source)!=anchor:
            raise ValueError('source tape differs')
        joint={side:rec['root_actions'] for side,rec in native.items()}
        if clean(joint['pay'][1-s])!=clean(joint['defer'][1-s]) or joint['pay'][s].get(u,'NONE')!='NONE':
            raise ValueError('unpaired simultaneous actions')
        nearby=[];intruders=[]
        for seat,observed in enumerate(start['players']):
            for uid,(position,cargo,role) in observed[2].items():
                if uid==u:continue
                distance=int(_ROUTE_DIST[position,cell])
                if distance>1:continue
                move=joint['pay'][seat].get(uid) or 'NONE'
                info={'seat':seat,'uid':uid,'role':role,'start':position,'cargo':cargo,'distance':distance}
                nearby.append(info)
                if int(_ROUTE_DEST[position,MOVES.index(move)])==cell:
                    intruders.append({**info,'executed_command':move})
        survival={}
        for side,commands in joint.items():
            env=begin(tape)
            for cmds in tape['actions'][:r['t']]:
                env.step(cmds)
            env.step(commands)
            p=env.states[0].observation.players
            survival[side]={who:who in p[seat][2] for who,seat in [(u,s)]+[(i['uid'],i['seat']) for i in intruders]}
            if survival[side][u]!=r[side]['root_worker_survived']:raise ValueError('next-step survival differs')
        rows.append({'training_map':r['seed'],'seat':s,'step':r['t'],'focal_uid':u,'cell':cell,
                     'public_before_action_reachable_workers':nearby,'observed_after_action_arrivals_if_wait':intruders,
                     'official_root_plus_one_surviving_ids':survival,'actually_paid':r['pay_intent_filled'],
                     'terminal_pay_minus_defer':r['target_delta_official_margin'],'source_actions_sha256':anchor,
                     'root_sha256':r['root_sha256']})
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('intakes',nargs='+',type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    pairs=[r for d in args.intakes for r in analyze(d)]
    if len(pairs)!=len({(r['training_map'],r['seat'],r['step'],r['focal_uid']) for r in pairs}):raise ValueError('duplicate roots')
    denied=[r for r in pairs if not r['actually_paid']]
    summary={'pairs':len(pairs),'actually_paid':len(pairs)-len(denied),'no_pay':len(denied),
             'denied_by_incoming_roles':Counter(entity['role'] for r in denied for entity in r['observed_after_action_arrivals_if_wait']),
             'unpaid_wait_dies_move_survives':sum(not r['official_root_plus_one_surviving_ids']['pay'][r['focal_uid']]
                                                  and r['official_root_plus_one_surviving_ids']['defer'][r['focal_uid']] for r in denied),
             'unpaid_contrasts_finishing_worse':sum(r['terminal_pay_minus_defer']<0 for r in denied)}
    retain(args.out,{'source_label_sha256':[sha(d/'training_labels.json') for d in args.intakes],'summary':summary,'rows':pairs})
    print(json.dumps(summary))

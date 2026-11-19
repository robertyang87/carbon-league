"""Select eligible pairs solely by map, physical seat and time; freeze prior tapes without reading their effects."""
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent
T=D.parent
sys.path.insert(0,str(T))
from select_roots import load,retain,sha

REG='bf8e5fc3c7dbf860a7ec02c6f1d65791a11886193208f41f8aaee20a10627657'


def choose():
    if sha(D/'protocol.json')!=REG:raise ValueError('selection protocol changed')
    end=T/'supplement_01/completion.json';completed=load(end)
    reg=load(D/'protocol.json');seen=[]
    for cohort in completed['counts']['source_intakes']:
        area=T/cohort['dir'];receipt=load(area/'source_receipt.json');selector=load(area/'selection.json')
        if sha(area/'training_labels.json')!=cohort['labels_sha256']:raise ValueError('incorrect frozen input')
        for label in load(area/'training_labels.json')['pairs']:
            if not label['pay_intent_filled']:continue
            r,=[r for r in selector['cases'] if (r['seed'],r['seat'],r['step'],r['uid'])==(label['seed'],label['seat'],label['t'],label['focal_uid'])]
            g,=[g for g in receipt['games'] if (g['seed'],g['seat'])==(r['seed'],r['seat'])]
            seen.append((area,r,g['map_sha256']))
    choices=[(a,b) for a in seen for b in seen if a[1]['seat']==0 and b[1]['seat']==1 and a[2]!=b[2]]
    def ordering(pair):
        return (-sum(r['step'] for _,r,_ in pair),*(val for _,r,_ in pair for val in (r['seed'],r['step'],r['uid'])))
    picked=min(choices,key=ordering);edges=[]
    if len(seen)!=33:raise ValueError('wrong frozen eligible count')
    for area,row,board in picked:
        old=area/row['case_directory']
        original={name:sha(old/name) for name in ('root.json.gz','training_label.json','live_protocol.json','live_shared_init.json',
                  'live_parent.json','live_alternative.json','live_parent_actions.json.gz','live_alternative_actions.json.gz')}
        edges.append({'area':str(area.relative_to(T)),'registered_case':row,'map_sha256':board,'old_files':original,
                      'new_stream_seeds':reg['streams'][f"seat{row['seat']}"]})
    packet={'protocol_sha256':REG,'completion_sha256':sha(end),'selection_program_sha256':sha(Path(__file__)),
            'eligible_steps':len(seen),'feasible_cross_seat_pairs':len(choices),'roots':edges}
    retain(D/'roots.json',packet);return packet


if __name__=='__main__':
    for r in choose()['roots']:
        g=r['registered_case']; print(g['seed'],g['seat'],g['step'],g['uid'],r['new_stream_seeds'])

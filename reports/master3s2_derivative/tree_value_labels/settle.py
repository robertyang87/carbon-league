"""Read-back checks for the two frozen forks. All margins are from our registered seat."""
from collections import Counter
from pathlib import Path
import gzip, hashlib, json
from register import HERE,sha,target

def read(path):
    data=path.read_bytes();return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)

def main():
    protocol=read(HERE/'protocol.json');rows=[]
    for ticket in protocol['tickets']:
        seed,s=ticket['seed'],ticket['seat'];r=read(HERE/f'{seed}_result.json')
        public=read(HERE/f'{seed}_root.json.gz');obs=public['public_root_by_seat'][s]
        assert r['root_sha256']==sha((HERE/f'{seed}_root.json.gz').read_bytes()) and ticket==public['ticket']
        arms={};tapes={}
        for name in ['stay','leave']:
            file=HERE/f'{seed}_{name}.json.gz';a=read(file)
            assert a['rows'][0]['actions'][s]==ticket[name] and a['root_enemy']==public['local_root_calls'][1-s][0]
            assert all(q['clocks'][i]['wall']>=0 for q in a['rows'] for i in (0,1))
            assert a['status']==['DONE','DONE'] and [q['step'] for q in a['rows']]==list(range(ticket['step'],a['final_t']))
            assert r['arms'][name]['sha256']==sha(file.read_bytes())
            root=a['rows'][0]; t_events=[e for e in root['flows'] if len(e)>4 and e[4]==ticket['cell']]
            on_cell=[dict(id=k,role=w[2],cargo=w[1],start=w[0],command=root['actions'][i].get(k,'NONE'),seat=i)
                     for i,p in enumerate(obs['players']) for k,w in p[2].items() if target(w[0],root['actions'][i].get(k,'NONE'))==ticket['cell']]
            focus_trees=[v['tree'] for v in root['tree_on_root_cell']]
            receipts=Counter()
            for step in a['rows']:
                for e in step['flows']:
                    if e[0]=='tree_absorption' and e[3] in focus_trees:receipts[e[1]]+=e[2]
            arms[name]=dict(scores=a['scores'],margin=a['official_margin'],focal_on_first=root['focal_planter'],root_flows=t_events,
               units_entering_focal=on_cell,tree_ids_after=root['tree_on_root_cell'],tree_gross_by_seat=[round(receipts[i],2) for i in (0,1)],
               steps=a['steps'],terminal=a['final_t'],slow=a['slow_calls'],hard=a['hard_calls'],max_wall=a['max_clock'],
               economics=a['flows_by_seat'],cpu_seconds=sum(c['cpu'] for q in a['rows'] for c in q['clocks']),
               H8_calls=[sum(q['clocks'][i]['H']==8 for q in a['rows']) for i in (0,1)],calls=[len(a['rows'])]*2)
            tapes[name]=a
        assert r['delta_stay_minus_leave']==round(arms['stay']['margin']-arms['leave']['margin'],2)
        diff=next((x['step'] for x,y in zip(tapes['stay']['rows'],tapes['leave']['rows']) if x['actions'][1-s]!=y['actions'][1-s]),None)
        rows.append(dict(ticket=ticket,arms=arms,delta_stay_minus_leave=r['delta_stay_minus_leave'],opponent_first_different_order_t=diff))
    path=HERE/'settlement.json'
    path.write_text(json.dumps(dict(protocol_sha256=sha((HERE/'protocol.json').read_bytes()),games=rows),sort_keys=True,indent=2,allow_nan=False)+'\n')
    print([(r['ticket']['seed'],r['delta_stay_minus_leave']) for r in rows])
if __name__=='__main__':main()

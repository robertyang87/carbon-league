"""First-tenure credit of each original late acquisition, in simulator event order."""
from pathlib import Path
from audit import HERE,ROOT,ev,sha,serial

def collect(game):
    raw=ROOT/game['source']
    assert ev.sha(raw)==game['raw_sha256']
    tape=ev.read(raw)
    env=ev.FastCarbon(tape['game_configuration']).reset(seed=game['seed'])
    s=game['seat'];active={};lots=[];rows={r['t']:r for r in game['rows']}
    for t,j in enumerate(tape['actions']):
        if t<279:
            env.step(*j)
            continue
        # A tree collision settles before the newly purchased ownership, even on a seize frame.
        step=ev.twin_step(env,j)
        for event in step['events']:
            channel,owner,amount,*keys=event
            if channel in ('tree_collision_cash','tree_absorption'):
                if channel=='tree_collision_cash':
                    tid=keys[2];cell=keys[1]
                else:tid=keys[0];cell=keys[1]
                lot=active.get(cell)
                if lot and lot['owner']==owner and lot['tid']==tid:
                    lot[channel]+=amount
            if channel in ('plant','seize'):
                worker,cell=keys
                if cell in active:
                    active.pop(cell)['exited_action']=t
                own_tree=[v for v in env.trees.values() if v.pos==cell and v.player==owner]
                if owner==s:
                    assert len(own_tree)==1
                    proposed=rows[t]['trace']['roots']
                    lot=dict(t=t,kind=channel,owner=owner,worker=worker,cell=cell,tid=own_tree[0].id,
                            cost=-amount,focal=bool(proposed and proposed[0]['uid']==worker),
                            tree_absorption=0.,tree_collision_cash=0.,exited_action=None)
                    lots.append(lot);active[cell]=lot
        owners={(tree.pos,tree.player,tree.id) for tree in env.trees.values()}
        for cell,lot in list(active.items()):
            if (cell,lot['owner'],lot['tid']) not in owners:
                active.pop(cell)['exited_action']=t
    for lot in lots:
        lot['first_tenure_net']=lot['tree_absorption']+lot['tree_collision_cash']-lot['cost']
    return {'seed':game['seed'],'seat':s,'raw_sha256':game['raw_sha256'],'lots':lots,'sum_paid':sum(r['cost'] for r in lots),
            'sum_gross':sum(r['tree_absorption']+r['tree_collision_cash'] for r in lots),'sum_net':sum(r['first_tenure_net'] for r in lots)}
if __name__=='__main__':
    d=ev.read(HERE/'diagnosis.json.gz');games=[collect(g) for g in d['games']]
    ev.write(HERE/'realized.json',{'input':ev.sha(HERE/'diagnosis.json.gz'),'code':ev.sha(Path(__file__)), 'games':games})
    for g in games:
        print(g['seed'],g['seat'],len(g['lots']),g['sum_paid'],g['sum_net'],[round(l['first_tenure_net'],1) for l in g['lots']])

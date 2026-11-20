"""Verify the beam honors fixed workers, reserved spawns and original move geometry."""
import ast
import numpy as np

from build import HERE,PARENT
names = {'SIZE','WACT','BACT','WIDX','BIDX','DR','DC','move','parse'}
defs = [n for n in ast.parse(PARENT.read_bytes()).body
        if isinstance(n,ast.FunctionDef) and n.name in names
        or isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names]
space = {'np':np}
exec(compile(ast.Module(body=defs,type_ignores=[]),str(PARENT),'exec'),space)
exec((HERE/'menu.py').read_text(),space)
beam = space['_joint_policy_beam']
players = [[500,{'b':0}, {'a':[1,70,'COLLECTOR'],'q':[16,25,'COLLECTOR'],'locked':[17,12,'PLANTER']}, {}],
           [80,{'e':90},{},{}]]
z = np.zeros((225,5),dtype=np.float32)
z[1,4] = 12
z[1,2] = 8
z[16,1] = 12
baseline = {'b':'RECPLANTER','locked':'UP'}
keys = {tuple(sorted(baseline.items()))}
results = beam(players,0,z,baseline,{'locked':'UP'},keys,3)
assert len(results)==len({tuple(sorted(order.items())) for order in results})==3
for order in results:
    assert all(order.get(uid)==action for uid,action in baseline.items())
    ends=[space['move'](w[0],space['WIDX'][order.get(uid,'NONE')]) for uid,w in players[0][2].items()]
    assert len(set(ends))==3 and 0 not in ends and 2 in ends
assert not beam(players,1,z,{},None,keys,3)
print('joint-width, uniqueness, reserved base and forced worker checks passed')

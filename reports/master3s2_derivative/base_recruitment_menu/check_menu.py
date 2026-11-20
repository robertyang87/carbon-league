"""Exercise the candidate's recruitment legality and occupied-square contract."""
import ast
import numpy as np

from build import HERE,PARENT
names = {'SIZE','WACT','BACT','WIDX','BIDX','DR','DC','move','parse'}
defs = [n for n in ast.parse(PARENT.read_bytes()).body
        if isinstance(n,ast.FunctionDef) and n.name in names
        or isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names]
space = {}
exec(compile(ast.Module(body=defs,type_ignores=[]),str(PARENT),'exec'),space)
exec((HERE/'menu.py').read_text(),space)
try_recruit = space['_recruitment_alternative']
base = {'r':0,'s':110}
enemy = [95,{'e':5}, {}, {}]
logits = np.zeros((225,3),dtype=np.float32)
logits[0,2] = 2.0
logits[110,2] = 1.0
players = [[30,base,{},{}],enemy]
assert try_recruit(players,0,logits,{},set()) == {'r':'RECPLANTER'}
assert try_recruit(players,0,logits,{}, {(('r','RECPLANTER'),)}) == {'s':'RECPLANTER'}
assert try_recruit(players,0,logits,{'s':'RECCOLLECTOR'},set()) == {'s':'RECPLANTER'}
players[0][0] = 29.99
assert try_recruit(players,0,logits,{},set()) is None
players[0][0] = 30
players[0][2] = {'w':[1,100,'COLLECTOR']}
assert try_recruit(players,0,logits,{'w':'LEFT'},set()) == {'w':'LEFT','s':'RECPLANTER'}
players[0][2] = dict.fromkeys(range(10),[1,100,'COLLECTOR'])
assert try_recruit(players,0,logits,{},set()) is None
print('6 candidate legality checks passed')

"""Verify the exact terminal callbacks against independently recorded official examples."""
import ast
import json

import numpy as np

from build import FAMILY, HERE, build, sha

digest=build()
raw=(HERE/'candidate.py').read_bytes()
assert sha(raw)==digest
tree=ast.parse(raw)
functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
search={node.name:node for node in functions['_search'].body if isinstance(node,ast.FunctionDef)}
check=dict(np=np,LAST_TURN=298,recruit_floor=30,me=0,op=1,settling=True,configuration={'episodeSteps':300})
wanted=[functions['_terminal'],*(search[name] for name in ('extinct','over','result'))]
exec(compile(ast.Module(body=wanted,type_ignores=[]),'candidate-terminal-callbacks','exec'),check)

source=(FAMILY/'sim_alignment/terminal_examples.json').read_bytes()
cases=json.loads(source)['cases']
observations=[]
for case in cases:
    old=case['root_cash']
    current=case['post_cash']
    # These registered cases all leave the seat-0 collector destroyed; case
    # index 3 also removes seat 1, and only case index 2 leaves seat 0 a tree.
    candidates=[
        [current[0],{'b0':0},{},{'tree':[112,3]} if case['has_tree'] else {}],
        [current[1],{'b1':2},{} if case['both'] else {'w':[185,91,'COLLECTOR']},{}],
    ]
    board=[0,0,dict(step=case['t']+1,players=candidates),[[old[0]],[old[1]]]]
    expected=case['official_terminal_value_if_done']
    assert check['over'](board)==(expected is not None)
    if expected is not None:
        assert check['result'](board)==expected
    observations.append(dict(t=case['t'],scores=case['official_rewards'],cash=current,expected=expected))
board=[0,0,dict(step=299,players=[[7000,{'b':0},{},{}],[6990,{'b':2},{},{}]]),[[6800],[6600]]]
assert check['over'](board) and check['result'](board)==check['_terminal'](10)
(HERE/'rules_check.json').write_text(json.dumps(dict(source=sha(source),candidate=digest,official_examples=observations,
                                                     full_horizon_uses_current_balance=True),indent=2)+'\n')

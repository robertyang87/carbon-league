"""Check the menu helper without loading any of the inference backends."""
import ast
import numpy as np

from build import HERE, PARENT

original = ast.parse(PARENT.read_bytes())
functions = {'move', 'dist', 'parse'}
constants = {'SIZE', 'WACT', 'WIDX', 'DR', 'DC'}
definitions = [
    node for node in original.body
    if isinstance(node, ast.FunctionDef) and node.name in functions
    or isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants
]
namespace = {'np': np}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(PARENT), 'exec'), namespace)
exec((HERE / 'menu.py').read_text(), namespace)
offer = namespace['_coordinated_return']
logits = np.zeros((225, 5), dtype=np.float32)
logits[1, 4] = 1.0
players = [
    [80, {'base': 0}, {'cargo': [2, 180, 'COLLECTOR'], 'blocker': [1, 0, 'PLANTER']}, {}],
    [80, {'enemy_base': 30}, {}, {}],
]
assert offer(players, 0, logits, {}, None) == {'cargo': 'LEFT', 'blocker': 'LEFT'}
assert offer(players, 0, logits, {'cargo': 'UP'}, None) == {'cargo': 'LEFT', 'blocker': 'LEFT'}
assert offer(players, 0, logits, {}, {'blocker': None}) is None
assert offer(players, 0, logits, {}, {'cargo': None}) is None
assert offer(players, 1, logits, {}, None) is None
logits[1, 2] = 2.0
assert offer(players, 0, logits, {}, None) == {'cargo': 'LEFT', 'blocker': 'RIGHT'}
assert offer(players, 0, logits, {'base': 'RECPLANTER'}, None) == {'base': 'RECPLANTER', 'cargo': 'LEFT', 'blocker': 'RIGHT'}
assert offer(players, 0, logits, {'cargo': 'LEFT', 'blocker': 'UP'}, None) is None
print('8 menu assertions passed')

"""Execute the real trainer's resume block with tiny Torch objects; no training."""
import ast
import json
import pathlib
import tempfile
import types
import warnings

import torch

torch.set_num_threads(1)
repo = pathlib.Path(__file__).resolve().parents[3]
path = repo / 'code/rl2_ppo_train.py'
tree = ast.parse(path.read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
block = next(n for n in main.body if isinstance(n, ast.If)
             and ast.unparse(n.test) == 'args.resume')
net = torch.nn.Linear(1, 1)
old = torch.optim.Adam(net.parameters(), lr=1e-5)
with tempfile.TemporaryDirectory(prefix='carbon-resume-review-') as tmp:
    ckpt = str(pathlib.Path(tmp) / 'state.pt')
    torch.save({'net': net.state_dict(), 'opt': old.state_dict(),
                'it': 24, 'kl_coef': 0.1}, ckpt)
    args = types.SimpleNamespace(resume=ckpt, lr=3e-5, kl=0.025)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ns = dict(torch=torch, args=args, net=net, opt=opt, vnet=None,
              it0=0, kl_coef=args.kl)
    code = compile(ast.Module(body=[block], type_ignores=[]), str(path), 'exec')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        exec(code, ns)
    result = {'source': str(path), 'source_line': block.lineno,
              'torch': torch.__version__, 'requested_lr': args.lr,
              'effective_lr': opt.param_groups[0]['lr'],
              'requested_kl': args.kl, 'effective_kl': ns['kl_coef']}
    assert result['effective_lr'] == 1e-5 and result['effective_kl'] == 0.1
    print(json.dumps(result))
rows = []
for ln, line in enumerate((repo / 'reports/live/snipe.jsonl').read_text().splitlines(), 1):
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if 'kl_coef' in row:
        rows.append((ln, row['it'], row['kl_coef']))
print(json.dumps({'real_log_rows': len(rows),
                  'iteration_min': min(r[1] for r in rows),
                  'iteration_max': max(r[1] for r in rows),
                  'effective_kl_values': sorted(set(r[2] for r in rows)),
                  'contradictory_change_record': 'reports/live/snipe.jsonl:55'}))

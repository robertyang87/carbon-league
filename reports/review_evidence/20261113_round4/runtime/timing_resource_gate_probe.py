"""One real 3.05 s call through the timing tool; tests its gate, not performance.
Only range(299) is shortened to one in memory; tool source remains untouched.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(os.environ['CARBON_REPO']).resolve()
OUT = Path(os.environ['CARBON_REVIEW_OUT']).resolve()

def worker():
    path = REPO / 'code/cnet2_search_timing.py'
    tree = ast.parse(path.read_text())
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'range' and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == 299):
            node.args[0].value = 1
            count += 1
    assert count == 1
    sys.argv = [str(path), str(OUT / 'slow_stub.py'), str(OUT / 'idle_stub.py'), '970500', '1']
    exec(compile(tree, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'slow_stub.py').write_text(
        'import time, types\n'
        '_AG = types.SimpleNamespace(stats={"searched": 0, "errors": 0})\n'
        'def agent(observation, configuration):\n'
        ' time.sleep(3.05)\n'
        ' return {}\n'
    )
    (OUT / 'idle_stub.py').write_text('def agent(observation, configuration):\n return {}\n')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    p = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'], capture_output=True, text=True, env=env)
    result = {'reviewed_sha': '5463f504048bcca28868ebce0a063ac8d6c51b8e',
              'method': 'real time.sleep(3.05); only timing range(299) -> range(1) AST edit',
              'not_a_model_performance_test': True, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    (OUT / 'timing_resource_gate_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    worker() if len(sys.argv) > 1 and sys.argv[1] == '--worker' else main()

"""Check replacement draws, including subsequent RNG state, for normalized categorical distributions."""
import ast
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    source = HERE/'candidate.py'
    tree=ast.parse(source.read_bytes())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_search_choice')
    context={'np':np}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),context)
    fast=context['_search_choice']
    generator=np.random.RandomState(82713)
    record={}
    for dtype in (np.float32,np.float64):
        for size in (3,5):
            cases=[np.eye(size,dtype=dtype)[i] for i in range(size)]
            for scale in (0,1,7,15,100):
                for _ in range(100):
                    logits=(scale*generator.normal(size=size)).astype(dtype)
                    probability=np.exp(logits-logits.max())
                    cases.append(probability/probability.sum())
            a=np.random.RandomState(17299)
            b=np.random.RandomState(17299)
            calls=0
            for p in cases:
                for _ in range(8):
                    assert a.choice(size,p=p)==fast(b,p)
                    calls+=1
            s,t=a.get_state(),b.get_state()
            assert s[0]==t[0] and np.array_equal(s[1],t[1]) and s[2:]==t[2:]
            record[f'{dtype.__name__}_{size}']=calls
    report=dict(numpy=np.__version__,draws=record,all_choices_and_final_states_equal=True)
    (HERE/f'rng_check_{np.__version__}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)

if __name__=='__main__':
    main()

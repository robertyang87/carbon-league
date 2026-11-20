from pathlib import Path
import types
p=Path('reports/master3s2_derivative/ladder/cargo_defense/development/hosted/cash_ledger.py').resolve()
source=p.read_text()
assert source.count('len(games) == len(keys) == 8')==1
source=source.replace('len(games) == len(keys) == 8','len(games) == len(keys) == 16').replace('cargo-defense-actual-game-economy/1','rollout-temperature-v2-actual-game-economy/1')
m=types.ModuleType('same_original_cash_ledger');m.__file__=str(p);exec(compile(source,str(p),'exec'),m.__dict__)
m.HERE=Path('reports/master3s2_derivative/rollout_temperature_v2/development/hosted').resolve()
m.main()

import json, os, sys
sys.path.insert(0, os.path.abspath('.'))
import eval_v0_1 as E

# one paired-side game: seed 1001, policy as P0 vs juzi, for a variant file
variant = sys.argv[1]
tag = sys.argv[2]
res = E.play_game((1001, 0, "reference_juzi", variant, tag))
print(json.dumps(res))

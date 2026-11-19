"""Favor immediately affordable planting on uncontested sites with a short estimated payback."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent = (HERE.parent / "planter_escape_full_dev/candidate.py").read_text()
marker = '        if apply_residual and slot < 10 and x["tok_kind"][slot] == 0:'
rule = '''        if slot < 10 and x["tok_kind"][slot] == 1 and x["left"] >= 30:
            pos = int(x["tok_pos"][slot])
            price = float(x["price"])
            free_site = not np.any(move[0, [0, 1, 5, 6, 8, 9]])
            if free_site and prefix.cash >= (prefix.plants + 1) * price + 30:
                neighbors = F.DEST[pos, 1:]
                estimated_flow = 3.75 * float(np.sum(aux["est"][neighbors]))
                if estimated_flow * 20 >= price:
                    combined[0] += 0.9
'''
assert parent.count(marker) == 1
(HERE / "candidate.py").write_text(parent.replace(marker, rule + marker))

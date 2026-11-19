"""Score an affordable seize or an adjacent approach with information available to the actor."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent = (HERE.parent / "planter_escape_full_dev/candidate.py").read_text()
marker = '        if apply_residual and slot < 10 and x["tok_kind"][slot] == 0:'
rule = '''        if slot < 10 and x["tok_kind"][slot] == 1 and prefix.cash >= 50:
            pos = int(x["tok_pos"][slot])
            for m in np.flatnonzero((move[:, 9] > 0) & (move[:, 0] == 0) & (move[:, 1] == 0)):
                cell = int(F.DEST[pos, m])
                age = round(float(x["maps"][18, cell]) * 50 / 255)
                horizon = min(16, 50 - age, x["left"]) - int(m != 0)
                flow = 3.75 * float(np.sum(aux["est"][F.DEST[cell, 1:]]))
                surplus = horizon * flow - 20.0
                if surplus > 0:
                    combined[m] += min(1.6, surplus / 35.0) * (1.0 if m == 0 else 0.65)
'''
assert parent.count(marker) == 1
(HERE / "candidate.py").write_text(parent.replace(marker, rule + marker))

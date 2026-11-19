"""Rebuild the frozen early planter escape candidate from the two-network parent."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "actor_partner_dev/candidate.py"
CANDIDATE = HERE / "candidate.py"


def build():
    parent = PARENT.read_text()
    marker = '        if apply_residual and slot < 10 and x["tok_kind"][slot] == 0:'
    assert parent.count(marker) == 1
    rule = '''        if slot < 10 and x["tok_kind"][slot] == 1 and x["left"] >= 199:
            ranked = np.where(legal, combined, -np.inf)
            preferred = int(np.argmax(ranked))
            safe = legal & (move[:, 0] == 0) & (move[:, 1] == 0)
            if move[preferred, 1] and np.any(safe):
                combined = combined + 1.0 * safe
'''
    CANDIDATE.write_text(parent.replace(marker, rule + marker))


if __name__ == "__main__":
    build()

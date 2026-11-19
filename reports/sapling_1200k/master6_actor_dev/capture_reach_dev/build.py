"""Price the second turn needed to finish a capture before directing a planter to approach."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE.parent/"planter_capture_full_dev/candidate.py").read_text()
marker='                    combined[m] += min(1.6, surplus / 35.0) * (1.0 if m == 0 else 0.65)'
rule='''                    # A move onto the tree only buys the option to seize on the following turn.
                    # move[:,1] already excludes adjacent enemies; examine the second ring here.
                    second_ring = F.DIST[cell] == 2
                    exposed = m != 0 and np.any(x["maps"][11:13, second_ring])
                    if exposed:
                        combined[m] -= 0.6
                    else:
                        combined[m] += min(1.6, surplus / 35.0) * (1.0 if m == 0 else 0.65)'''
assert source.count(marker)==1
(HERE/"candidate.py").write_text(source.replace(marker,rule))

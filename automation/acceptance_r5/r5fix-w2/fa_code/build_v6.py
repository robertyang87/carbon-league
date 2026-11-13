"""Dataset v6: v5 teachers with slot 3 re-assigned from 1C-Zero-v7 to the AgyCarbon-P2C-Godslayer family
(versions XC..XCIV, all rated >= 2480; one shared style slot), per-unit decision weights (CNET2_UW=1).
slots: 0 Master, 1 1C-Zero-v9, 2 v8-preview, 3 Godslayer-P2C, 4 RyanX-V11, 5 Master2, 6 1C-Zero-v10, 7 v11-preview.
WB-* strategies are never teachers (held-out exam family); 1C-Zero-v7 sides are dropped.
  CNET2_WEIGHTED=1 CNET2_UW=1 python3 build_v6.py OUT DIR [DIR...]"""
import sys
import cnet2_feat as F

GODSLAYER = {"AgyCarbon-P2C-Godslayer-" + v for v in ("XC", "XCI", "XCII", "XCIII", "XCIV")}
SLOT3 = "AgyCarbon-P2C-Godslayer"


class _Teachers(list):
    def _m(self, n):
        return SLOT3 if n in GODSLAYER else n

    def __contains__(self, n):
        return list.__contains__(self, self._m(n))

    def __getitem__(self, i):
        r = list.__getitem__(self, i)
        return _Teachers(r) if isinstance(i, slice) else r

    def index(self, n, *a):
        return list.index(self, self._m(n), *a)


T = _Teachers(["claude-AlphaCarbon-Master", "1C-Zero-v9", "1C-Zero-v8-preview", SLOT3,
               "RyanX-AIPowercore-V11 Ultra", "claude-AlphaCarbon-Master2", "1C-Zero-v10", "1C-Zero-v11-preview", "other"])
F.TEACHERS = T
import cnet2_build as B  # noqa: E402
B.F.TEACHERS = T
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

"""Held-out stand-in dataset: ONLY 无始6.0 / 无始7.0 sides (teacher slot 6). python3 build_ws.py OUT DIR
Slot 6 is warm-started as-is from the base net (no BC_REINIT_TEACH): unlike the Godslayer slot in v6, the style
vector starts from whatever slot 6 meant in the init weights and is fine-tuned from there."""
import sys
import cnet2_feat as F


class _T(list):
    def _m(self, n):
        return "无始" if n in ("无始6.0", "无始7.0") else n

    def __contains__(self, n):
        return list.__contains__(self, self._m(n))

    def __getitem__(self, i):
        r = list.__getitem__(self, i)
        return _T(r) if isinstance(i, slice) else r

    def index(self, n, *a):
        return list.index(self, self._m(n), *a)


T = _T(["_t0", "_t1", "_t2", "_t3", "_t4", "_t5", "无始", "other"])
F.TEACHERS = T
import cnet2_build as B  # noqa: E402
B.F.TEACHERS = T
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

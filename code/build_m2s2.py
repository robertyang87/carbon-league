"""Held-out stand-in v2 for Master2s: ALL its ladder games, mapped to slot 5 (the Master2 slot of the v7 table),
so it can warm-start from b128v7 (stronger base than b128). python3 build_m2s2.py OUT DIR"""
import sys
import cnet2_feat as F
F.TEACHERS[:] = ["_t0", "_t1", "_t2", "_t3", "_t4", "claude-AlphaCarbon-Master2s", "_t6", "_t7", "other"]
import cnet2_build as B
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

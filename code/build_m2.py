"""Build a BC dataset of ONLY claude-AlphaCarbon-Master2's sides (teacher slot 6) for a held-out test stand-in.
  python3 build_m2.py OUT_DIR REPLAY_DIR"""
import sys
import cnet2_feat as F
F.TEACHERS[:] = ["_t0", "_t1", "_t2", "_t3", "_t4", "_t5", "claude-AlphaCarbon-Master2", "other"]
import cnet2_build as B
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

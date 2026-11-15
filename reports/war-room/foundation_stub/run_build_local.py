"""Local build_top-equivalent validation: build_top.py overwrites
cnet2_feat.TEACHERS from CNET2_TOP_TEACHERS ("name@row") before calling
cnet2_build.main. Replicated here: m3s2r1 at row 2 (index 2), 'other' last
(scan keeps TEACHERS[:-1]; teacher_id falls back to 'other')."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnet2_feat as F
F.TEACHERS = ["local-pad-0", "local-pad-1", "m3s2r1", "other"]
import cnet2_build as B
B.main(sys.argv[1], sys.argv[2:])

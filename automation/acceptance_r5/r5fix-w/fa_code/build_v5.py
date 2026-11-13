"""Dataset v5: top-bot BC with the NEW strong teachers as their own style slots.
slots: 0 Master, 1 1C-Zero-v9, 2 v8-preview, 3 v7, 4 RyanX-V11, 5 Master2, 6 1C-Zero-v10, 7 1C-Zero-v11-preview.
Only listed teachers' sides are kept (no 'other' side is ever used).  python3 build_v5.py OUT DIR [DIR...]"""
import sys
import cnet2_feat as F
F.TEACHERS[:] = ["claude-AlphaCarbon-Master", "1C-Zero-v9", "1C-Zero-v8-preview", "1C-Zero-v7",
                 "RyanX-AIPowercore-V11 Ultra", "claude-AlphaCarbon-Master2", "1C-Zero-v10", "1C-Zero-v11-preview", "other"]
import cnet2_build as B
if __name__ == "__main__":
    B.main(sys.argv[1], sys.argv[2:])

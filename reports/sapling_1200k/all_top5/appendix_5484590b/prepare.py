"""Archive all new appearances above the six non-overlapping manifests that contain 1,373 maps."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from appendix_batch import prepare


if __name__ == "__main__":
    manifests = [HERE.parent / "manifest.json",
                 HERE.parent / "appendix_499886f1/manifest.json",
                 HERE.parent / "appendix_4bfdabf5/manifest.json",
                 HERE.parent / "appendix_555c0c56/manifest.json",
                 HERE.parent / "appendix_bf049043/manifest.json",
                 HERE.parent / "appendix_bd3c0021/manifest.json"]
    prepare("5484590b3fae60b3c906c6a0be9f7840c241388a", manifests, HERE)

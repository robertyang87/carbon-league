"""Archive all new appearances above the seven non-overlapping manifests that contain 1,378 maps."""
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
                 HERE.parent / "appendix_bd3c0021/manifest.json",
                 HERE.parent / "appendix_5484590b/manifest.json"]
    prepare("99919cbf5889caa98303091e146aa210fa035d18", manifests, HERE, expected=(4, 0))

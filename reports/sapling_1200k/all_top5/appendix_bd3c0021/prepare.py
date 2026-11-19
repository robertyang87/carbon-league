"""Archive all new appearances above the five non-overlapping manifests that contain 1,361 maps."""
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
                 HERE.parent / "appendix_bf049043/manifest.json"]
    prepare("bd3c0021b0992e900236df9dbdf5cd05774ee11e", manifests, HERE)

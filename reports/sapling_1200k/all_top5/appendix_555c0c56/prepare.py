"""Archive all new appearances above the three non-overlapping manifests that contain 1,347 maps."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from appendix_batch import prepare


if __name__ == "__main__":
    manifests = [HERE.parent / "manifest.json",
                 HERE.parent / "appendix_499886f1/manifest.json",
                 HERE.parent / "appendix_4bfdabf5/manifest.json"]
    prepare("555c0c566bd2572032568daeb8fa79f73a4c36e7", manifests, HERE)

"""Archive all new appearances above the four non-overlapping manifests that contain 1,357 maps."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from appendix_batch import prepare


if __name__ == "__main__":
    manifests = [HERE.parent / "manifest.json",
                 HERE.parent / "appendix_499886f1/manifest.json",
                 HERE.parent / "appendix_4bfdabf5/manifest.json",
                 HERE.parent / "appendix_555c0c56/manifest.json"]
    prepare("bf049043a995807e723f5345c39469aaf59050de", manifests, HERE)

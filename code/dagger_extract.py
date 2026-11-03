#!/usr/bin/env python3
"""dagger_extract -- turn dagger traces into a corpus-schema npz
(bc_data/dagger_corpus.npz) using extract_bc_v53's own sampler, so
the DAgger samples are byte-compatible with bc_v53_corpus rows."""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_bc as X  # noqa: E402
import extract_bc_v53 as E  # noqa: E402


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] \
        if "--tag" in sys.argv else "dagger"
    manifest = json.loads((HERE / f"{tag}_manifest.json").read_text())
    counters = Counter()
    samples = []
    for g in manifest:
        with gzip.open(HERE / "traces" / g["file"], "rt",
                       encoding="utf-8") as fh:
            trace = json.load(fh)
        got = E.samples_from_game(g, trace, counters)
        samples.extend(got)
        print(f"{g['file']}: {len(got)} samples")
    packed = X.pack(samples)
    out = HERE / "bc_data" / f"{tag}_corpus.npz"
    np.savez_compressed(out, **packed)
    print(f"games={len(manifest)} samples={len(samples)} "
          f"counters={dict(counters)}")
    print("wrote", out)


if __name__ == "__main__":
    main()

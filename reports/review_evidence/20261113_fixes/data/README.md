# Dataset and split re-review evidence

Reviewed repository commit: `14d53320b20d58bdb82676f7fa866b8e11a55062`.

## Run

Use a Python environment with the repository's runtime dependencies (including NumPy):

```bash
CARBON_REPO=/path/to/carbon-league \
CARBON_REVIEW_OUT=/path/to/review-output \
python recheck_data_round3.py
```

`CARBON_REPO` takes priority. When it is unset, the script walks upward from its
own location to find `code/fastsim.py`. Results always print to stdout;
`CARBON_REVIEW_OUT` optionally names the directory receiving
`data-round3-results.json`. All generated datasets and injected input failures
live in a temporary directory and are removed when the run finishes. Business
source files and remote services are not modified.

The replay fixture contains only three decisions, with an index length of 300
to pass the scanner. This deliberately isolates cache behavior; it is not a
complete arena replay. The actual builder and merger execute without function
mocks.

## Observed result

- Original-game split inheritance passed. Legacy corpus shards lacking a usable
  `orig_id` were rejected before output was written; new self-play shards worked.
- Configuration, teacher mapping, score, and order changes invalidated caches.
- Empty-window rebuilds remained empty on retry; stale shard files were deleted.
- Failed gzip reads did not certify stale shards; retry after repairing the input
  restored the complete dataset. The previous failure-cache defect passed.
- A separate partial-publication issue remains: a failed source game still lets
  the builder exit 0 and publish a partial dataset. With one corrupt game out of
  two, metadata retained two games with row counts `[0, 3]`. Both `build_top.py`
  and `set_uw.py` exited 0, and the replica chain's game-count check passed.

The script ran only the real pre-training build and weighting stages. It did
**not** run `replica_chain.sh`, launch training, or start remote jobs. The later
training-launch consequence is established by the repository call chain, not
by launching a training task. See `replica_prelaunch_gates` in the JSON evidence.

The saved JSON records the reviewed commit. Re-running against another commit
may legitimately produce different observations.

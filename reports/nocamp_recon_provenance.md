# nocamp_recon provenance

- Requested original: `claude-opus-v53-nocamp`, author `cjc_de_ds`, local census id `aa758d5a42687ff7aff528cf694095c9`.
- Local leaderboard/census snapshots mark this strategy `is_public: false`; a direct source fetch attempt through the available browser fetch tool was blocked by policy for the site domain, and per instruction that denied fetch was not retried through another route.
- Therefore this benchmark uses the fallback authorised by the task: a local reconstruction from `base_v53.py`.
- `base_v53.py` is byte-identical to `reference_claude_opus_v53.py` (md5 `1779b453672f5b85d6df1cf473e64ab0`).
- Parameter existence/original values were verified in `base_v53.py`: `soft_w=1.0` (line 274) and `campers=4, camp_from=6` (line 282).
- Applied exactly: `soft_w 1.0 -> 0.0`, `campers 4 -> 1`, `camp_from 6 -> 12`; functional diff is archived in `nocamp_recon.diff` (plus a provenance header comment in the .py file).

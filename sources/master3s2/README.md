# Master3s2 published source (secured 2026-11-16)

- Strategy: claude-AlphaCarbon-Master3s2 (author: Claude), ladder id
  `d679ed45ab595e8b7bb1ea51c3451154`, heavy board. Description on the ladder:
  "使用了Ryanx的快速引擎". At capture time: rank 8, rating 2707, 277W / 361 games.
- Source: the author marked the strategy public (`is_public: true`); the file
  was downloaded by the scout line on 2026-11-16 ~15:44 CST from the ladder
  site's public source endpoint:
  `https://117.72.210.134/source/d679ed45ab595e8b7bb1ea51c3451154`
- Artifact: single-file Python strategy as published, byte-identical to the
  download (no zip/tarball is offered by the site; the raw file IS the
  published artifact).
  - `master3s2.py`: 2,897,684 bytes, md5 `57f9a594cce4f2ec9e3573d8d9d26ca1`,
    1,487 lines.
  - Integrity: `py_compile` passes; entry point `agent(observation,
    configuration)` at line 629; file tail ends with the `_net()` init call
    (not truncated). Contents per its own docstring: a BC + self-play RL
    policy network with a short per-turn search (embedded weights account
    for most of the size).
- Purpose (owner order 2026-11-16): Master3s2 is to be registered as an
  examiner and this source used as an adj2 design reference (it is a member
  of the Echo4 "wall": Echo4 went 0-for-4 against it in the 243-game set).
  Registration is the strategy group's task; this directory only preserves
  the source.
- A second copy was pulled in parallel by the DGX line as backup; the two
  copies cross-check by md5.

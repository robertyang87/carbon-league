# S1-2 A1–A3 acceptance tests

Fixtures (machine-side, not in repo): a working dir containing
`replays/` (20 m3s2 corpus games, `*.json.gz`) and `heads_v2.npz`
(canonical heads, md5 `06a03d9cfeec239af3065a49fefec382`). Copy the code
files under test plus these tests into that dir and run with a torch
venv (A1 needs torch; A2/A3 are numpy-only).

- `test_s12_a1.py` — A1: numpy `act(want_u=True)` u vs torch
  `forward_heads` utility (tol 1e-5), u differs from old value head,
  missing utility keys raise KeyError.
- `test_s12_a2.py` — A2: teacher terminal U vs eval_v53 F01/F04 on hand
  cases (extinct win/loss, timeout) and 20 real terminal states
  (tol 1e-9); `_extinct_state` mirror check.
- `test_s12_a3.py` — A3: search-agent guard parity before/after the
  BankruptcyGuard extraction (63 positions, writes
  guard_parity_old/new.json), plus synthetic trigger-branch cases.
- `test_s12_a3_teacher.py` — A3 teacher side: CachePolicy vs NP.Policy
  parity with guard off (must be exact), guard-on smoke over 50 root
  positions, guard on/off differences reported (not judged).

First run: 2026-11-13, Mac Studio (torch 2.3.1, CPU), against the
S1 wiring commit. A1/A2/A3 PASS (see reports/STAGE1_TEACHER_GATES_20261113.md §1).

# Spot Replacement Bootstrap — Runbook

Purpose: when an Aliyun exam spot (spot-29 / spot-30 class) is reclaimed,
stand up a replacement that is tailnet-joined, key-authorized, and loaded
with the golden exam stack — with no interactive approval steps. The only
secret involved is a reusable Tailscale auth key, supplied at runtime via
secure entry (never in chat, never committed, never hardcoded).

All timings below are ESTIMATES, extrapolated from the one measured build
on 2026-11-13 (see "Measured facts"), not guarantees.

## Code version anchor (owner order, 2026-11-13 13:23)

- **Code baseline: carbon-league `main` HEAD
  `a7ed8f9552ffc7217c8ea4fe432d575713b4295f`.** Wherever the bootstrap /
  deploy flow touches repository code, it must use this HEAD (and, for
  future replacements, the then-current `main` HEAD), never the code
  files inside the 2026-11-13 11:21 golden image. The golden image
  supplies only the environment (`.venv`), `data/weights` and
  `data/packs`; its `code/` files are stale (its bigeval / certify
  predate the Codex round-5 fixes).
- After extracting the golden image, refresh the code files on the
  target from the anchor HEAD and verify these md5s (read back from the
  GitHub API at the anchor commit on 2026-11-13; `main` tip confirmed
  equal to the anchor):
  - `code/eval_v53.py` `aa4c5bb63c879caffeb33b0b25c498d7` (unchanged
    from the golden manifest)
  - `code/bigeval.py` `327ff8073932cc8bd59c3e59090d2647` (golden had
    `3547eb6a…`, superseded)
  - `code/certify_pack.py` `3c058ddc219272c389c4061907338b41` (new
    check; not in the golden manifest)
  - `code/cnet2_search.py` `10a0674e5fd572801d2a06876f1de54b` (the
    `14230fa1…` value quoted in older notes and in the 2026-11-13
    engineering task text is stale at this anchor; the API read-back
    at the anchor governs)
- Reference copies of these four files, byte-identical to the anchor,
  are staged on spot-31 at `/root/code_refresh_a7ed8f95/` (verified
  on-machine 2026-11-13 13:24) and kept locally at
  `~/workspace/carbon-cloud-shanghai/code_refresh_a7ed8f95/`.
  `deploy_stack.sh` applies this staging directory after extraction
  when present, and its manifest check enforces the anchor md5s, so a
  deploy that only uses golden-image code now fails loudly instead of
  passing stale code.

## Roles

- **C3 line** — buys the replacement machine (standing rule, no per-incident
  approval): same region (Shanghai N) and spec class as the lost machine,
  price gate ≤ ¥2.582141/hr, fleet target 2 exam spots / floor 1, never a
  third. C3 also delivers bootstrap.sh to the new machine (user-data at
  purchase, or console delivery right after) and reports the new machine's
  tailnet IP (from BOOTSTRAP-READY) to the war room.
- **War room** — mints the bootstrap auth key on demand through the
  Tailscale API (custom.tailscale connection, verified 2026-11-13) and
  hands it to C3 for delivery. The owner's one-time setup behind this is
  done: an API access token stored in the connection, plus a reusable
  auth key in his password manager as the manual fallback.
- **DGX line** — runs deploy_stack.sh ON DGX (as arm, using DGX's local
  carbon_relay private key) against the tailnet IP C3 reports, verifies the
  manifest, and signs EXAM-READY to the war room. C3 does NOT run
  deploy_stack.sh (it has no DGX arm access); the war room coordinates and
  hands the ready machine to its exam line (C1/C2).

## Procedure

1. **Trigger.** A spot is reclaimed (C3 patrol / side-chat report notices
   the machine gone). Any exam lines on it are already protected by the
   2-minute off-machine sync rule; nothing else needs rescuing.

2. **Buy (C3).** Purchase the replacement per the standing rule above.
   Note its public IP for step 3 delivery. (Estimate: ~10 min.)

3. **Bootstrap (on the new machine, as root).** Deliver `bootstrap.sh` to
   the new machine via the Aliyun console (cloud assistant / user-data)
   and run it with the auth key in the environment only:
   `TS_AUTH_KEY=<minted by the war room via the Tailscale API> SPOT_NAME=carbon-spot-<n> bash bootstrap.sh`
   It installs Tailscale, joins the tailnet non-interactively, authorizes
   the hatch + DGX carbon_relay public keys, and prints `BOOTSTRAP-READY`
   plus the machine's tailnet IPv4. Note that IPv4 for step 4.
   (Estimate: ~3–5 min.)

4. **Deploy the stack (from DGX, as arm).**
   `bash deploy_stack.sh <new-machine-tailnet-ipv4>`
   It verifies the golden tarball's md5 against its sidecar, pushes the
   tarball over the tailnet with the carbon_relay key, extracts it to
   `/root/carbon`, then verifies the manifest on the target (venv python
   imports numpy; md5 of `code/eval_v53.py`, `code/bigeval.py`,
   `code/cnet2_search.py` match the spot-30 originals; `data/weights`
   present). Success prints `EXAM-READY`; any failure aborts loudly.
   (Estimate: ~10–15 min; the 2.0 GB push dominates.)

5. **Report.** The deploying line sends the war room: machine identity
   (instance id, tailnet IPv4, hostname), the `BOOTSTRAP-READY` and
   `EXAM-READY` lines, and the golden md5 used. The war room re-assigns
   the exam queue. (Total replacement-to-exam-ready: ~30 min, ESTIMATE.)

## Golden image

- Lives on DGX: `/home/arm/junyang/golden/exam_stack_golden.tar.zst`
  (sidecar `exam_stack_golden.md5`; build script `make_golden.sh`).
- Facts are recorded in `golden.md5` next to this file: md5
  `fdd877ba3bb69a5bc84c62027c227180`, 2,018,439,231 bytes compressed /
  2,532,065,280 uncompressed, 4,536 entries (4,191 files).
- Snapshot of spot-30's `/root/carbon` taken 2026-11-13 (code, `code/exp`,
  `.venv`, `data/weights` ×186, `data/packs`; runtime outputs excluded).
  If the exam stack changes materially, rebuild on DGX with
  `make_golden.sh` and refresh `golden.md5` before relying on this package.

## Measured facts (2026-11-13, the build this package is based on)

- spot-30 `/root/carbon` = 2.5 GB: code 936 MB (of which `code/exp` 918 MB),
  data 1.4 GB (weights 1.4 GB / 186 files, packs 8.8 MB), `.venv` 229 MB.
- DGX→spot-30 tailnet stream (tar, nice 19 on source, zstd on DGX) ran
  ~8 min at ~4–5 MB/s. Exams running on spot-30 were not disturbed.
- Manifest md5s were verified three ways: on spot-30, inside the tarball
  after extraction on DGX, and against the values in `deploy_stack.sh`.

## GAPS / not yet proven

- **Auth key supply — CLOSED 2026-11-13 via the API mint route.** The
  war room mints a fresh auth key per incident through the Tailscale API
  (devices read, mint and delete loop all verified 2026-11-13; the first
  production mint went to the spot-31 replacement). Fallbacks if the API
  route is down: the owner's password-manager reusable key (one paste),
  or the old manual approval link.
- **No full dry-run.** Steps 3→4 have not been executed against a real
  fresh instance (no spare machine exists to rehearse on without buying
  one). Components were validated piecemeal: key authorization (proven
  2026-11-13 on both current spots), relay-key login from DGX (proven),
  tarball integrity + manifest extraction (proven on DGX). The spot-31
  replacement (bought 2026-11-13 after spot-30 was reclaimed) is the
  first full run, under way now; whoever runs it should watch each step's
  output rather than fire-and-forget.
- **Snapshot age.** The golden image freezes the stack as of 2026-11-13.
  A replacement restored from it runs that stack, not any later fixes;
  reconcile with the repo's current main after EXAM-READY if the stack
  has moved since.
- **Proxy quirk (war-room VM only).** Parallel SSH sessions through this
  VM's egress proxy failed once with a broken pipe during package setup;
  sequential sessions are reliable. Drive steps 3–4 sequentially.

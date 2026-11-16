"""Equivalence-gate harness (PREREG_PERGAME_LOGGING_20261116 section 2.2).

The pinned trainer seeds its league-sampling RNG from wall-clock time
(rl2_ppo_train.py: rng = default_rng(int(time.time()) % 100000 + it0)), so two
ordinary runs can never be bit-identical, with or without any patch. This
harness pins time.time() to a fixed epoch BEFORE the trainer starts, making
the registered fixed-seed on/off comparison meaningful. It changes no trainer
code; production runs never use it.

Usage: gate_pin_time.py <trainer_path> [trainer args...]
"""
import runpy
import sys
import time

_FIXED = 1700000000.0
time.time = lambda: _FIXED  # noqa: E731 - deliberate pin for the gate harness

trainer = sys.argv[1]
sys.argv = [trainer] + sys.argv[2:]
runpy.run_path(trainer, run_name="__main__")

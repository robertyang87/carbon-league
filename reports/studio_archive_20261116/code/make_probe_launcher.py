"""Build the Studio probe launcher copy from launch_ppou2_v7.py.

Only the filed changes (see ops-room filing):
  1. --device cuda -> mps (sole functional change)
  2. path remap to the probe dir (venv/cwd/W/OUT/LOG/launch.json)
  3. env: drop PYTORCH_CUDA_ALLOC_CONF, add PYTORCH_ENABLE_MPS_FALLBACK=1
No config values change. Original file untouched.
"""
import pathlib

P = pathlib.Path("/Users/arm/junyang/carbon/data/ppou2_probe_studio")
src = P / "launch_ppou2_v7.py"
dst = P / "launch_ppou2_v7_probe.py"
t = src.read_text()
orig = t

assert '"--device", "cuda"' in t
t = t.replace('"--device", "cuda"', '"--device", "mps"')
assert '"/root/carbon/.venv/bin/python"' in t
t = t.replace('"/root/carbon/.venv/bin/python"',
              '"/Users/arm/junyang/carbon/.venv/bin/python"')
assert 'OUT = f"/root/carbon/data/rl3/ppou2_{ARM}"' in t
t = t.replace('OUT = f"/root/carbon/data/rl3/ppou2_{ARM}"',
              'OUT = "/Users/arm/junyang/carbon/data/ppou2_probe_studio/out"')
assert 'LOG = f"/root/carbon/data/rl3/ppou2_{ARM}.log"' in t
t = t.replace('LOG = f"/root/carbon/data/rl3/ppou2_{ARM}.log"',
              'LOG = "/Users/arm/junyang/carbon/data/ppou2_probe_studio/out/ppou2_probe.log"')
assert 'W = "/root/carbon/data/weights"' in t
t = t.replace('W = "/root/carbon/data/weights"',
              'W = "/Users/arm/junyang/carbon/data/ppou2_probe_studio/weights"')
assert 'cwd="/root/carbon/code"' in t
t = t.replace('cwd="/root/carbon/code"',
              'cwd="/Users/arm/junyang/carbon/data/ppou2_probe_studio/code"')
assert 'f"/root/carbon/data/rl3/ppou2_{ARM}.launch.json"' in t
t = t.replace('f"/root/carbon/data/rl3/ppou2_{ARM}.launch.json"',
              '"/Users/arm/junyang/carbon/data/ppou2_probe_studio/out/ppou2_probe.launch.json"')
assert '"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",' in t
t = t.replace('    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",\n', '')
assert '"RL2_SERVER_MODELS": "20",' in t
t = t.replace('"RL2_SERVER_MODELS": "20",',
              '"RL2_SERVER_MODELS": "20", "PYTORCH_ENABLE_MPS_FALLBACK": "1",')
assert t != orig
dst.write_text(t)
print("WROTE", dst)

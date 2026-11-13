"""F04 certify_pack counterexamples on the final tree (real subprocesses)."""
import json, os, subprocess, sys, tempfile
CODE = os.path.expanduser("~/workspace/final-accept/code")
PY = os.path.expanduser("~/workspace/integ-accept/venv/bin/python")
tmp = tempfile.mkdtemp(prefix="cert-fa-")
env = dict(os.environ, CNET2_NO_ROLL="1")
env.pop("CNET2_ENS", None)

def run(name, src, extra, pack_env=None):
    p = os.path.join(tmp, name + ".py")
    if src is not None: open(p, "w").write(src)
    out = os.path.join(tmp, name + ".cert.json")
    cmd = [PY, "-B", os.path.join(CODE, "certify_pack.py"), p, *extra, "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=pack_env or env)
    cert = None
    try: cert = json.loads(r.stdout)
    except Exception: pass
    return r.returncode, cert, r.stderr[-200:]

res = []
def chk(name, cond, info=""):
    res.append(cond); print(("PASS " if cond else "FAIL ") + name, info)

LEGAL = "def agent(obs,cfg): return {}\n"
rc, c, err = run("legal", LEGAL, ["--steps-per-game", "2"])
chk("legal PASS", rc == 0 and c and c["overall"] == "PASS", f"rc={rc} overall={c and c['overall']}")

COLD = ("import time\ncalled=False\ndef agent(obs,cfg):\n global called\n"
        " if not called: time.sleep(3.05); called=True\n return {}\n")
rc, c, err = run("cold_first", COLD, ["--steps-per-game", "2"])
fails = [k for k, v in (c or {}).get("checks", {}).items() if v.get("pass") is False] if c else []
chk("first-call-slow FAIL", c and c["overall"] == "FAIL", f"overall={c and c['overall']} failing={fails}")

LAST = ("import time\nn=[0]\ndef agent(obs,cfg):\n n[0]+=1\n"
        " if n[0]==2: time.sleep(3.05)\n return {}\n")
rc, c, err = run("last_slow", LAST, ["--steps-per-game", "2"])
g = (c or {}).get("games", [{}])[0]
chk("last-advance-slow FAIL", c and c["overall"] == "FAIL",
    f"overall={c and c['overall']} first_call_ms={g.get('first_call_ms')} step_max={g.get('step_wall_ms',{}).get('max')} game_wall={g.get('game_wall_s')}")

WARM = "import time\ndef agent(obs,cfg):\n time.sleep(3.05)\n return {}\n"
rc, c, err = run("every_step_slow", WARM, ["--steps-per-game", "1"])
chk("every-step-slow FAIL", c and c["overall"] == "FAIL", f"overall={c and c['overall']}")

TWO = "import time\ndef agent(obs,cfg):\n time.sleep(1.02)\n return {}\n"
rc, c, err = run("two_games", TWO, ["--steps-per-game", "2", "--games", "2"])
chk("two-games-2-slow-each PASS", rc == 0 and c and c["overall"] == "PASS",
    f"rc={rc} overall={c and c['overall']} slow={c and c['aggregate'].get('slow_steps_over_soft')} pool={c and c['aggregate'].get('overage_pool_s')}")

missing = os.path.join(tmp, "nonexistent-opponent.py")
rc, c, err = run("bad_opp", LEGAL, ["--steps-per-game", "2", "--opp", missing])
chk("missing opponent nonzero exit", rc != 0, f"rc={rc} err={err[:80]!r}")

n1r = os.path.join(CODE, "packs", "n1r_int8_reference.py")
out = os.path.join(tmp, "n1r.cert.json")
r = subprocess.run([PY, "-B", os.path.join(CODE, "certify_pack.py"), n1r,
                    "--steps-per-game", "2", "--mem-limit-mib", "512", "--out", out],
                   capture_output=True, text=True, timeout=180, env=env)
c = json.loads(r.stdout) if r.stdout.strip().startswith("{") else None
chk("n1r short PASS", r.returncode == 0 and c and c["overall"] == "PASS",
    f"rc={r.returncode} overall={c and c['overall']} a_errors={c and c['aggregate'].get('a_errors')} peak_rss={c and c['aggregate'].get('peak_rss_mib_child_self')}")
print(f"== F04: {sum(res)}/{len(res)} ==")
sys.exit(0 if all(res) else 1)

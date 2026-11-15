import json, os, signal, subprocess, sys, time

RUN = '/root/carbon/data/rl3/snipe_n1'
TRAIN = RUN + '/train.jsonl'
STATE = RUN + '/state_last.pt'
LOG = '/root/carbon/data/rl3/snipe_n1.log'
STATUS = RUN + '/codeswitch_p1fix_status.txt'
SRC = '/root/carbon/code_next/'
DST = '/root/carbon/code/'
TARGETS = {
    DST + 'cnet2_model.py': '3bb5857f833a94797ac6cdc2a5a28963',
    DST + 'cnet2_train.py': '39e749cb177e3d72511e8d9fc986da80',
    DST + 'cnet2_build.py': 'd6e867187ceec289c0e28a53399eb01a',
    DST + 'producer/selfplay_gen.py': '95d2e29fbec8cfcebee57bf9bf0d25ed',
}
DEADLINE = time.time() + 90 * 60

def log(msg):
    with open(STATUS, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')

def last_it():
    rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
    return rows[-1]['it']

def main_pid():
    out = subprocess.run(['pgrep', '-f', 'rl2_ppo_train.py'], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None

def md5(path):
    return subprocess.run(['md5sum', path], capture_output=True, text=True).stdout.split()[0]

def main():
    open(STATUS, 'w').write('')
    log('WATCHER_START waiting for it64')
    base_it = last_it()
    while time.time() < DEADLINE:
        if not main_pid():
            log('ALERT trainer not running; no action')
            return
        if last_it() > base_it:
            break
        time.sleep(30)
    else:
        log('TIMEOUT no new iteration; no action')
        return
    pid = main_pid()
    boundary_it = last_it()
    log('SWITCH_START boundary_it=%d pid=%d' % (boundary_it, pid))
    argv = [a for a in open('/proc/%d/cmdline' % pid, 'rb').read().decode().split('\0') if a]
    env = {}
    for kv in open('/proc/%d/environ' % pid, 'rb').read().decode().split('\0'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            env[k] = v
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not main_pid():
            break
        time.sleep(3)
    if main_pid():
        log('ABORT trainer did not exit')
        return
    log('TRAINER_STOPPED')
    bak = RUN + '/state_pre_p1fix_it%d.pt' % boundary_it
    subprocess.run(['cp', STATE, bak], check=True)
    log('STATE_BACKUP %s md5 %s' % (bak, md5(bak)))
    r = subprocess.run(['rsync', '-a', SRC, DST], capture_output=True, text=True)
    if r.returncode != 0:
        log('ABORT rsync rc=%d %s' % (r.returncode, r.stderr[-300:]))
        return
    log('CODE_OVERLAY_DONE')
    for path, want in TARGETS.items():
        got = md5(path)
        if got != want:
            log('ABORT md5 mismatch %s got %s want %s; NOT relaunching' % (path, got, want))
            return
    log('MD5_ALL_MATCH')
    logsize = os.path.getsize(LOG)
    with open(LOG, 'a') as lf:
        p = subprocess.Popen(argv, cwd='/root/carbon/code', env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    log('RELAUNCHED pid=%d' % p.pid)
    time.sleep(30)
    alive = main_pid()
    log('POSTCHECK30 alive_pid=%s' % alive)
    time.sleep(90)
    alive2 = main_pid()
    log('POSTCHECK120 alive_pid=%s last_it=%d' % (alive2, last_it()))
    try:
        with open(LOG, 'rb') as lf:
            lf.seek(logsize)
            newlog = lf.read().decode('utf-8', 'replace')
        hits = [l for l in newlog.split('\n') if any(k in l.lower() for k in ('migrat', 'weight_gate', 'traceback', 'error'))]
        for h in hits[:12]:
            log('LOGHIT ' + h[:240])
    except Exception as e:
        log('LOGREAD_FAIL %r' % e)
    log('SWITCH_DONE' if alive2 else 'RELAUNCHED_BUT_EXITED')

main()

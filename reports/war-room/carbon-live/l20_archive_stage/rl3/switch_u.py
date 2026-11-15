import json, os, signal, subprocess, time

RUN = '/root/carbon/data/rl3/snipe_n1'
TRAIN = RUN + '/train.jsonl'
EVAL = RUN + '/eval.jsonl'
STATE = RUN + '/state_last.pt'
LOG = '/root/carbon/data/rl3/snipe_n1.log'
STATUS = RUN + '/switch_u_status.txt'
STAGED = '/root/carbon/code_next/rl2_ppo_train.py'
LIVE = '/root/carbon/code/rl2_ppo_train.py'
EXPECT = '92e97652206452cc2020050c13114066'
GO = '/root/carbon/data/rl3/termU_GO'
DEADLINE = time.time() + 72 * 3600

def log(msg):
    with open(STATUS, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')

def md5(path):
    return subprocess.run(['md5sum', path], capture_output=True, text=True).stdout.split()[0]

def last_it():
    rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
    return rows[-1]['it']

def main_pid():
    out = subprocess.run(['pgrep', '-f', 'rl2_ppo_train.py'], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None

def trainer_argv(pid):
    return [a for a in open('/proc/%d/cmdline' % pid, 'rb').read().decode().split('\0') if a]

def pool_has_r1(pid):
    try:
        argv = trainer_argv(pid)
        return 'm9xj320r1' in argv[argv.index('--pool') + 1]
    except Exception:
        return False

def eval_landed(eit):
    if not os.path.exists(EVAL):
        return False
    its = [json.loads(l)['it'] for l in open(EVAL) if l.strip()]
    return its.count(eit) >= 2

def set_arg(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    return argv

def main():
    open(STATUS, 'w').write('')
    log('WATCHER_START termU gates: GO-flag + pool-r1 + eval-guard; staged md5 %s' % md5(STAGED))
    if md5(STAGED) != EXPECT:
        log('ABORT staged md5 mismatch at arm time')
        return
    base = None
    while time.time() < DEADLINE:
        pid = main_pid()
        if not pid:
            log('ALERT trainer not running; no action')
            return
        ready = os.path.exists(GO) and pool_has_r1(pid)
        if ready:
            li = last_it()
            e = (li // 5) * 5
            if e >= 70 and not eval_landed(e) and li < e + 3:
                ready = False
        if ready:
            if base is None:
                base = last_it()
                log('GATES_PASS base_it=%d; firing at next boundary' % base)
            elif last_it() > base:
                break
        time.sleep(45)
    else:
        log('TIMEOUT gates never passed; no action')
        return
    pid = main_pid()
    boundary_it = last_it()
    log('SWITCH_START boundary_it=%d pid=%d' % (boundary_it, pid))
    argv = trainer_argv(pid)
    env = {}
    for kv in open('/proc/%d/environ' % pid, 'rb').read().decode().split('\0'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            env[k] = v
    argv_old_holder = list(argv)
    argv = set_arg(argv, '--reward', 'u')
    argv = set_arg(argv, '--rscale', '1.0')
    argv = set_arg(argv, '--gamma', '1.0')
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not main_pid():
            break
        time.sleep(3)
    if main_pid():
        log('ABORT trainer did not exit')
        return
    log('TRAINER_STOPPED')
    bak = RUN + '/state_pre_termU_it%d.pt' % boundary_it
    subprocess.run(['cp', STATE, bak], check=True)
    log('STATE_BACKUP %s md5 %s' % (bak, md5(bak)))
    fbak = RUN + '/rl2_ppo_train.py.pre_termU'
    subprocess.run(['cp', LIVE, fbak], check=True)
    subprocess.run(['cp', STAGED, LIVE], check=True)
    if md5(LIVE) != EXPECT:
        subprocess.run(['cp', fbak, LIVE], check=True)
        log('ABORT live md5 mismatch after copy; old file restored, relaunching old argv')
        with open(LOG, 'a') as lf:
            subprocess.Popen(argv_old_holder, cwd='/root/carbon/code', env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        return
    log('FILE_SWAP_DONE live md5 %s' % md5(LIVE))
    with open(LOG, 'a') as lf:
        p = subprocess.Popen(argv, cwd='/root/carbon/code', env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    log('RELAUNCHED pid=%d reward=u rscale=1.0 gamma=1.0' % p.pid)
    time.sleep(30)
    log('POSTCHECK30 alive_pid=%s' % main_pid())
    time.sleep(90)
    log('POSTCHECK120 alive_pid=%s last_it=%d' % (main_pid(), last_it()))
    log('SWITCH_DONE')

main()

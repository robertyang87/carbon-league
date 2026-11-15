import json, os, signal, subprocess, sys, time

RUN = '/root/carbon/data/rl3/snipe_n1'
TRAIN = RUN + '/train.jsonl'
EVAL = RUN + '/eval.jsonl'
STATE = RUN + '/state_last.pt'
LOG = '/root/carbon/data/rl3/snipe_n1.log'
STATUS = RUN + '/poolswap_320r1_status.txt'
NEW_ENTRY = ["SNAP:cnet2:/root/carbon/data/weights/cnet2_m9xj320r1.npz#T2", 0.5]
EVAL_IT = 70
DEADLINE = time.time() + 4 * 3600

def log(msg):
    with open(STATUS, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')

def last_it():
    rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
    return rows[-1]['it']

def eval_done():
    if not os.path.exists(EVAL):
        return False
    its = [json.loads(l)['it'] for l in open(EVAL) if l.strip()]
    return its.count(EVAL_IT) >= 2

def main_pid():
    out = subprocess.run(['pgrep', '-f', 'rl2_ppo_train.py'], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None

def md5(path):
    return subprocess.run(['md5sum', path], capture_output=True, text=True).stdout.split()[0]

def main():
    open(STATUS, 'w').write('')
    log('WATCHER_START eval-protect it%d' % EVAL_IT)
    it_at_eval = None
    while time.time() < DEADLINE:
        if not main_pid():
            log('ALERT trainer not running; no action')
            return
        if eval_done():
            if it_at_eval is None:
                it_at_eval = last_it()
                log('EVAL_LANDED at it=%d; waiting next boundary' % it_at_eval)
            elif last_it() > it_at_eval:
                break
        time.sleep(45)
    else:
        log('TIMEOUT waiting for eval/boundary; no action')
        return
    pid = main_pid()
    boundary_it = last_it()
    log('SWAP_START boundary_it=%d pid=%d' % (boundary_it, pid))
    argv = [a for a in open('/proc/%d/cmdline' % pid, 'rb').read().decode().split('\0') if a]
    env = {}
    for kv in open('/proc/%d/environ' % pid, 'rb').read().decode().split('\0'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            env[k] = v
    i = argv.index('--pool')
    pool = json.loads(argv[i + 1])
    if any('m9xj320r1' in e[0] for e in pool):
        log('ABORT pool already contains m9xj320r1')
        return
    pool.append(NEW_ENTRY)
    argv[i + 1] = json.dumps(pool)
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not main_pid():
            break
        time.sleep(3)
    if main_pid():
        log('ABORT trainer did not exit')
        return
    log('TRAINER_STOPPED')
    bak = RUN + '/state_pre_pool320r1_it%d.pt' % boundary_it
    subprocess.run(['cp', STATE, bak], check=True)
    log('STATE_BACKUP %s md5 %s' % (bak, md5(bak)))
    logsize = os.path.getsize(LOG)
    with open(LOG, 'a') as lf:
        p = subprocess.Popen(argv, cwd='/root/carbon/code', env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    log('RELAUNCHED pid=%d pool_members=%d' % (p.pid, len(pool)))
    time.sleep(30)
    log('POSTCHECK30 alive_pid=%s' % main_pid())
    time.sleep(90)
    log('POSTCHECK120 alive_pid=%s last_it=%d' % (main_pid(), last_it()))
    log('SWAP_DONE')

main()

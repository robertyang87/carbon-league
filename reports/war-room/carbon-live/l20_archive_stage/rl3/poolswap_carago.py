import json, os, signal, subprocess, sys, time

RUN = '/root/carbon/data/rl3/snipe_n1'
TRAIN = RUN + '/train.jsonl'
EVAL = RUN + '/eval.jsonl'
STATE = RUN + '/state_last.pt'
LOG = '/root/carbon/data/rl3/snipe_n1.log'
STATUS = RUN + '/poolswap_carago_status.txt'
NEW_ENTRY = ["SNAP:cnet2:/root/carbon/data/weights/cnet2_caragor1_ep19.npz#T2", 0.5]
DEADLINE = time.time() + 3 * 3600

def log(msg):
    with open(STATUS, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')

def last_it():
    rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
    return rows[-1]['it']

def eval60_done():
    if not os.path.exists(EVAL):
        return False
    its = [json.loads(l)['it'] for l in open(EVAL) if l.strip()]
    return its.count(60) >= 2

def main_pid():
    out = subprocess.run(['pgrep', '-f', 'rl2_ppo_train.py'], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None

open(STATUS, 'w').write('')
log('WATCHER_START')
it_at_eval = None
while time.time() < DEADLINE:
    pid = main_pid()
    if not pid:
        log('ALERT trainer not running; watcher exits without action')
        sys.exit(0)
    if eval60_done():
        if it_at_eval is None:
            it_at_eval = last_it()
            log('EVAL60_LANDED at it=%d; waiting next boundary' % it_at_eval)
        elif last_it() > it_at_eval:
            break
    time.sleep(45)
else:
    log('TIMEOUT waiting for eval60/boundary; no action taken')
    sys.exit(0)

pid = main_pid()
boundary_it = last_it()
log('SWAP_START boundary_it=%d pid=%d' % (boundary_it, pid))
argv = open('/proc/%d/cmdline' % pid, 'rb').read().decode().split('\0')
argv = [a for a in argv if a]
env = {}
for kv in open('/proc/%d/environ' % pid, 'rb').read().decode().split('\0'):
    if '=' in kv:
        k, v = kv.split('=', 1)
        env[k] = v
i = argv.index('--pool')
pool = json.loads(argv[i + 1])
if any('caragor1' in e[0] for e in pool):
    log('ABORT pool already contains caragor1')
    sys.exit(0)
pool.append(NEW_ENTRY)
argv[i + 1] = json.dumps(pool)
os.kill(pid, signal.SIGTERM)
for _ in range(60):
    if not main_pid():
        break
    time.sleep(3)
if main_pid():
    log('ABORT trainer did not exit after SIGTERM')
    sys.exit(0)
log('TRAINER_STOPPED')
bak = RUN + '/state_pre_poolcarago_it%d.pt' % boundary_it
subprocess.run(['cp', STATE, bak], check=True)
md5 = subprocess.run(['md5sum', bak], capture_output=True, text=True).stdout.split()[0]
log('STATE_BACKUP %s md5 %s' % (bak, md5))
with open(LOG, 'a') as lf:
    p = subprocess.Popen(argv, cwd='/root/carbon/code', env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
log('RELAUNCHED pid=%d pool_members=%d' % (p.pid, len(pool)))
time.sleep(20)
alive = subprocess.run(['pgrep', '-f', 'rl2_ppo_train.py'], capture_output=True, text=True).stdout.split()
log('POSTCHECK trainer_pids=%s last_it=%d' % (','.join(alive), last_it()))
log('SWAP_DONE')

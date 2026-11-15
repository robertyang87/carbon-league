import gh_api, time
MARK = "R1FABLE1520X9K"
RID = 43174815464
t0 = time.time()
deadline = time.time() + 300
status = None
while time.time() < deadline:
    try:
        r = gh_api.req('GET', f'/repos/robertyang87/carbon-league/actions/runs/{RID}')
        status = r['status']
        print('poll', time.strftime('%H:%M:%S'), status, r.get('conclusion'), flush=True)
        if status == 'completed':
            break
    except Exception as e:
        print('poll error:', e, flush=True)
    time.sleep(20)
print('final status', status, 'elapsed', round(time.time()-t0), flush=True)
try:
    jobs = gh_api.req('GET', f'/repos/robertyang87/carbon-league/actions/runs/{RID}/jobs')
    for j in jobs.get('jobs', []):
        print('JOB', j['id'], j['status'], j.get('conclusion'), flush=True)
        try:
            log = gh_api.get_job_logs('robertyang87', 'carbon-league', j['id'])
        except Exception as e:
            print('log error:', e, flush=True); continue
        if isinstance(log, str) and MARK in log:
            i0 = log.find(MARK)
            print(log[max(0, i0-200):i0+12000], flush=True)
            print('MARKER-CLAIMED', flush=True)
        elif isinstance(log, str):
            print('log lines', len(log.splitlines()), 'no marker; tail:', flush=True)
            for l in log.splitlines()[-25:]:
                print(' ', l[:300], flush=True)
        else:
            print('log type', type(log), flush=True)
except Exception as e:
    print('jobs error:', e, flush=True)
print('HARVEST-DONE', flush=True)

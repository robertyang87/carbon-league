"""Poll ladder-replay-pull run 43296592007 (m9xj460 precache) to completion,
then count data/m9xj460_corpus files on main. Writes result JSON to
m9xj460_precache_result.json next to this script. Max ~18 minutes."""
import sys, json, time, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

RUN_ID = 43296592007
OUT = '/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/m9xj460_precache_result.json'
deadline = time.time() + 18 * 60
status = conclusion = None
while time.time() < deadline:
    d = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{RUN_ID}")
    status = d.get('status')
    conclusion = d.get('conclusion')
    if status == 'completed':
        break
    time.sleep(90)

result = {"run_id": RUN_ID, "status": status, "conclusion": conclusion}
if status == 'completed' and conclusion == 'success':
    # count corpus files via git tree (recursive, main)
    tree = gh_api.req("GET", "/repos/robertyang87/carbon-league/git/trees/main?recursive=1")
    files = [t['path'] for t in tree.get('tree', []) if t['path'].startswith('data/m9xj460_corpus/') and t['type'] == 'blob']
    result["corpus_files"] = len(files)
    result["sample"] = files[:3]
json.dump(result, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(result, ensure_ascii=False))

"""Transfer the original Actions artifact bytes to their existing research lane."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
RUN = 44720618161
COMMIT = '46da42cf581fa533ad2da81640c394f2b9c42190'
URL = 'repos/robertyang87/carbon-league/actions/'
PROTOCOL = '89570d213782d5fee2bc2e33f572f3c622d286393f605846c45ca5f90d262619'


def get(endpoint):
    return subprocess.check_output(['gh', 'api', URL + endpoint], stderr=subprocess.DEVNULL)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    run = json.loads(get(f'runs/{RUN}'))
    assert run['head_sha'] == COMMIT and run['run_attempt'] == 1 and run['status'] == 'completed'
    raw = (HERE / 'protocol.json').read_bytes()
    assert sha(raw) == PROTOCOL
    expected = {f"strategic-dev-{row['seed']}-{row['seat']}-{RUN}" for row in json.loads(raw)['schedule']}
    listed = json.loads(get(f'runs/{RUN}/artifacts?per_page=100'))
    assert listed['total_count'] == len(listed['artifacts'])
    chosen = [a for a in listed['artifacts'] if a['name'] in expected]
    assert len(chosen) == len(expected) == 8 and {a['name'] for a in chosen} == expected
    out = HERE / 'hosted'
    out.mkdir(exist_ok=True)
    for item in chosen:
        assert item['workflow_run']['id'] == RUN and not item['expired']
        dest = out / f"{item['id']}.zip"
        data = get(f"artifacts/{item['id']}/zip")
        assert len(data) == item['size_in_bytes'] and item['digest'] == 'sha256:' + sha(data)
        if dest.exists():
            assert dest.read_bytes() == data
        else:
            dest.write_bytes(data)
        with zipfile.ZipFile(dest) as archive:
            assert archive.testzip() is None
        print(item['id'], item['name'], item['digest'])


if __name__ == '__main__':
    main()

import sys, hashlib, urllib.request
sys.path.insert(0, '/home/hatch/workspace')
sys.path.insert(0, '/opt/muse/skills/custom-connector-setup/bin')
from dynamic_credentials import add_surrogate_to_request

path, out = sys.argv[1], sys.argv[2]
url = f'https://api.github.com/repos/robertyang87/carbon-league/contents/{path}?ref=main'
req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github.raw'})
add_surrogate_to_request(req, 'custom.github', allowed_hosts=['api.github.com'])

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

opener = urllib.request.build_opener(NoRedirect)
try:
    resp = opener.open(req, timeout=60)
    data = resp.read()
    open(out, 'wb').write(data)
    print('bytes:', len(data))
    print('md5:', hashlib.md5(data).hexdigest())
    sys.exit(0)
except urllib.request.HTTPError as e:
    loc = e.headers.get('Location')
    if not loc:
        print('ERROR: no Location, status', e.code); sys.exit(1)
data = urllib.request.urlopen(loc, timeout=120).read()
open(out, 'wb').write(data)
print('bytes:', len(data))
print('md5:', hashlib.md5(data).hexdigest())

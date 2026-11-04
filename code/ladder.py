"""Carbon League 天梯客户端（标准库，无依赖）。

用法（在 code/ 下）：
  python3 ladder.py ping                         # 连通性（不登录）
  python3 ladder.py login <用户名>                # 密码从环境变量 LADDER_PASS 或交互输入读取，不落盘
  python3 ladder.py whoami                       # 当前登录身份 + 今日剩余提交次数
  python3 ladder.py submit <文件> <策略名> [说明文件]  # 说明默认取文件头 docstring
  python3 ladder.py mine                         # 我的策略（含定位进度、榜外排名）
  python3 ladder.py board [N]                    # 排行榜前 N（默认 30）
  python3 ladder.py matches <策略ID> [输出.json]   # 拉该策略全部对局（compact，分页）
  python3 ladder.py replays <matches.json> <目录>  # 下载回放（登录后；≤30 次/分钟；已存在跳过）

环境变量：
  LADDER_BASE     默认 https://132.232.176.153 。云容器里裸 IP 的 HTTPS 会被出网代理重置（无 SNI），
                  先起 `python3 sni_bridge.py`，再设 LADDER_BASE=http://127.0.0.1:18443（Host 头自动设为 IP）。
                  sslip.io 之类域名不可用：腾讯云对未备案域名拦截。
  LADDER_COOKIES  会话 cookie 文件，默认 ~/.carbon_ladder_cookies（仓库外，不要提交）。
证书：服务器是自签名证书，与旧脚本 curl -k 一致不校验。
"""
import ast, gzip, http.cookiejar, json, os, ssl, sys, time, uuid, urllib.error, urllib.request

BASE = os.environ.get("LADDER_BASE", "https://132.232.176.153").rstrip("/")
HOST = os.environ.get("LADDER_HOST", "132.232.176.153" if "127.0.0.1" in BASE else "")
COOKIES = os.path.expanduser(os.environ.get("LADDER_COOKIES", "~/.carbon_ladder_cookies"))

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE
_jar = http.cookiejar.LWPCookieJar(COOKIES)
if os.path.exists(COOKIES):
    _jar.load(ignore_discard=True, ignore_expires=True)
_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler(),  # 读取 HTTPS_PROXY
    urllib.request.HTTPSHandler(context=_ctx),
    urllib.request.HTTPCookieProcessor(_jar),
)
_csrf = None


def _save():
    _jar.save(ignore_discard=True, ignore_expires=True)
    os.chmod(COOKIES, 0o600)


def request(path, method="GET", body=None, headers=None, raw=False, timeout=60):
    h = {"Accept": "application/json"}
    if HOST:
        h["Host"] = HOST
    if _csrf:
        h["X-CSRF-Token"] = _csrf
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=h)
    try:
        with _opener.open(req, timeout=timeout) as r:
            data = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        data, status = e.read(), e.code
    _save()
    if raw:
        return status, data
    try:
        js = json.loads(data)
    except ValueError:
        raise RuntimeError(f"{method} {path}: HTTP {status}, 非 JSON 响应: {data[:200]!r}")
    if status >= 400:
        raise RuntimeError(f"{method} {path}: HTTP {status}: {js.get('error', js)}")
    return js


def session():
    global _csrf
    s = request("/api/session")
    _csrf = s.get("csrf")
    return s


def cmd_ping():
    t = time.time()
    d = request("/api/leaderboard")
    print(f"OK {BASE}  {len(d.get('strategies', []))} 条榜单  {time.time() - t:.2f}s")


def cmd_login(user):
    pw = os.environ.get("LADDER_PASS")
    if not pw:
        import getpass
        pw = getpass.getpass(f"{user} 的密码: ")
    session()
    request("/api/login", "POST", {"username": user, "password": pw})
    cmd_whoami()


def cmd_whoami():
    s = session()
    u = s.get("user")
    print(f"登录: {u.get('name')}  今日剩余提交: {s.get('remaining')}" if u else "未登录")


def docstring_of(path):
    try:
        return ast.get_docstring(ast.parse(open(path, encoding="utf-8").read())) or ""
    except SyntaxError:
        return ""


def cmd_submit(path, name, desc_path=None):
    src = open(path, "rb").read()
    assert 0 < len(src) <= 10_000_000, f"文件大小 {len(src)} 超限"
    desc = open(desc_path, encoding="utf-8").read() if desc_path else docstring_of(path)
    s = session()
    if not s.get("user"):
        sys.exit("未登录：先 python3 ladder.py login <用户名>")
    print(f"提交 {path} ({len(src):,} B) 为「{name}」，今日剩余 {s.get('remaining')} 次")
    b = uuid.uuid4().hex
    parts = []
    for k, v in (("name", name), ("description", desc)):
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="submission.py"\r\n'
                 f'Content-Type: text/x-python; charset=utf-8\r\n\r\n'.encode() + src + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    r = request("/api/submit", "POST", b"".join(parts),
                {"Content-Type": f"multipart/form-data; boundary={b}"}, timeout=300)
    print(json.dumps(r, ensure_ascii=False, indent=1)[:2000])
    cmd_whoami()


def cmd_mine():
    session()
    for s in request("/api/mine").get("strategies", []):
        keep = {k: v for k, v in s.items() if not isinstance(v, (dict, list)) and k != "description"}
        print(json.dumps(keep, ensure_ascii=False))


def cmd_board(n=30):
    d = request("/api/leaderboard")
    rows = d.get("strategies", [])[: int(n)]
    for i, s in enumerate(rows, 1):
        print(f"{i:3d} {s.get('name', '?')[:34]:34s} {s.get('rating', 0):7.0f} ±{s.get('rating_sd', 0):3.0f} "
              f"{s.get('wins', 0):3d}/{s.get('games', 0):<3d} {s.get('board', ''):5s} {s.get('ranking_phase', '')}  {s.get('id', '')}")
    if not rows:
        print(json.dumps(d, ensure_ascii=False)[:1500])


def cmd_matches(sid, out=None):
    got, off = [], 0
    while True:
        d = request(f"/api/matches?compact=1&limit=100&offset={off}&strategy={sid}")
        got += d.get("matches", [])
        if d.get("next_offset") is None:
            break
        off = d["next_offset"]
        time.sleep(0.7)
    seen, uniq = set(), []
    for m in got:
        k = (m.get("id"), m.get("game"))
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    out = out or f"matches_{sid}.json"
    json.dump(uniq, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"{len(uniq)} 局 -> {out}")


def cmd_replays(mfile, outdir):
    session()
    os.makedirs(outdir, exist_ok=True)
    ms = json.load(open(mfile))
    n = 0
    for m in ms:
        if m.get("winner") is None and not m.get("scores"):
            continue  # 未结束
        fn = os.path.join(outdir, f"{m['id']}-{m['game']}.json.gz")
        if os.path.exists(fn):
            continue
        st, data = request(f"/data/{m['id']}/{m['game']}", raw=True, timeout=120)
        if st != 200:
            print(f"{m['id']}-{m['game']}: HTTP {st}")
        else:
            if data[:2] != b"\x1f\x8b":
                data = gzip.compress(data)
            open(fn, "wb").write(data)
            n += 1
        time.sleep(2.1)  # ≤30 次/分钟
    print(f"新下载 {n} 局 -> {outdir}")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmds = {"ping": cmd_ping, "login": cmd_login, "whoami": cmd_whoami, "submit": cmd_submit,
            "mine": cmd_mine, "board": cmd_board, "matches": cmd_matches, "replays": cmd_replays}
    if not a or a[0] not in cmds:
        sys.exit(__doc__)
    cmds[a[0]](*a[1:])

"""本地明文→TLS 桥：绕开「云容器出网代理不接受无 SNI 的 TLS」的问题。

背景（2026-11-04 实测）：
- 天梯只有裸 IP 132.232.176.153；curl / Python ssl 对 IP 地址不发 SNI，出网代理遇到无 SNI 的 ClientHello 直接重置。
- 用域名（sslip.io 等）带 SNI 会被腾讯云未备案拦截（HTTP 302 到 dnspod webblock，HTTPS 重置）。
- 把 SNI 设成 IP 字符串本身（openssl s_client -servername 132.232.176.153）两边都放行。

本脚本监听 127.0.0.1:18443（明文），每个连接经 HTTPS_PROXY 做 CONNECT 到目标:443，
再用 pyOpenSSL（可设任意 SNI）建 TLS，双向搬字节。不校验证书（与旧脚本 curl -k 一致，服务器为自签名）。
  pip install pyopenssl
  nohup python3 sni_bridge.py >/tmp/sni_bridge.log 2>&1 &
  LADDER_BASE=http://127.0.0.1:18443 python3 ladder.py ping
"""
import os, select, socket, sys, threading, urllib.parse
from OpenSSL import SSL

TARGET = os.environ.get("BRIDGE_TARGET", "132.232.176.153")
LISTEN = int(os.environ.get("BRIDGE_PORT", "18443"))
px = urllib.parse.urlparse(os.environ["HTTPS_PROXY"])


def upstream():
    s = socket.create_connection((px.hostname, px.port), timeout=30)
    s.sendall(f"CONNECT {TARGET}:443 HTTP/1.1\r\nHost: {TARGET}:443\r\n\r\n".encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        c = s.recv(4096)
        if not c:
            raise OSError("proxy closed during CONNECT")
        buf += c
    if b" 200 " not in buf.split(b"\r\n", 1)[0]:
        raise OSError(f"CONNECT refused: {buf[:200]!r}")
    ctx = SSL.Context(SSL.TLS_CLIENT_METHOD)
    ctx.set_verify(SSL.VERIFY_NONE, lambda *a: True)
    ctx.set_alpn_protos([b"http/1.1"])
    t = SSL.Connection(ctx, s)
    t.set_tlsext_host_name(TARGET.encode())
    t.set_connect_state()
    s.settimeout(None)
    t.do_handshake()
    s.setblocking(False)
    return t


def handle(c):
    """单线程 select 循环：TLS 对象只被一个线程读写（OpenSSL 连接不支持并发读写，大上传会卡死）。"""
    try:
        t = upstream()
    except Exception as e:
        print("upstream error:", e, flush=True)
        c.close()
        return
    raw = t.get_socket() if hasattr(t, "get_socket") else None
    raw = raw or t._socket
    up_buf = b""          # 待发往服务器的明文
    client_open = True
    try:
        while True:
            rl = [raw] + ([c] if client_open and len(up_buf) < (1 << 20) else [])
            wl = [raw] if up_buf else []
            r, w, _ = select.select(rl, wl, [], 300)
            if not r and not w and not t.pending():
                break
            if c in r:
                d = c.recv(65536)
                if not d:
                    client_open = False
                else:
                    up_buf += d
            if up_buf and (raw in w or raw in r):
                try:
                    n = t.send(up_buf[:16384])
                    up_buf = up_buf[n:]
                except (SSL.WantWriteError, SSL.WantReadError):
                    pass
            if raw in r or t.pending():
                while True:
                    try:
                        d = t.recv(65536)
                    except (SSL.WantReadError, SSL.WantWriteError):
                        break
                    except (SSL.ZeroReturnError, SSL.SysCallError):
                        return
                    if not d:
                        return
                    c.sendall(d)
                    if not t.pending():
                        break
            if not client_open and not up_buf:
                pass  # 等服务器把响应发完再关
    except Exception as e:
        print("relay error:", repr(e), flush=True)
    finally:
        c.close()
        try:
            t.close()
        except Exception:
            pass


def main():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", LISTEN))
    srv.listen(16)
    print(f"bridge 127.0.0.1:{LISTEN} -> {TARGET}:443 (SNI={TARGET}) via {px.hostname}:{px.port}", flush=True)
    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()

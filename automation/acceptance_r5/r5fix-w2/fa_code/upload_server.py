import http.server, os, sys, datetime

OUT = os.path.expanduser("~/workspace/carbon-league/replays/alpha2_full")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.expanduser("~/workspace/carbon-league/upload_server.log")

def log(msg):
    line = f"{datetime.datetime.now().isoformat()} {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

class H(http.server.BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_GET(self):
        if self.path.startswith("/ping"):
            self.send_response(200); self._cors(); self.end_headers()
            self.wfile.write(b"pong")
            log(f"PING from {self.client_address}")
        else:
            self.send_response(404); self._cors(); self.end_headers()
    def do_POST(self):
        if self.path.startswith("/upload/"):
            name = os.path.basename(self.path[len("/upload/"):].split("?")[0])
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            with open(os.path.join(OUT, name), "wb") as f:
                f.write(body)
            log(f"UPLOAD {name} {len(body)} bytes from {self.client_address}")
            self.send_response(200); self._cors()
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(('{"saved":"%s","bytes":%d}' % (name, len(body))).encode())
        else:
            self.send_response(404); self._cors(); self.end_headers()
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8901
    log(f"START port {port}")
    http.server.HTTPServer(("0.0.0.0", port), H).serve_forever()

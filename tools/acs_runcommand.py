#!/usr/bin/env python3
"""Minimal Aliyun ECS Cloud Assistant runner (RPC signature v1, stdlib only).

Env: ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, ACS_REGION (default cn-shanghai)
Args: <instance_id> <command|--b64 base64_command>
Polls DescribeInvocationResults until the invocation finishes, prints decoded output.
"""
import base64, hashlib, hmac, json, os, sys, time, urllib.parse, urllib.request, uuid

AK = os.environ["ALIYUN_ACCESS_KEY_ID"]
SK = os.environ["ALIYUN_ACCESS_KEY_SECRET"]
REGION = os.environ.get("ACS_REGION", "cn-shanghai")
HOST = f"https://ecs.{REGION}.aliyuncs.com/"


def pct(s):
    return urllib.parse.quote(str(s), safe="~")


def call(action, **params):
    p = {
        "Action": action,
        "Version": "2014-05-26",
        "Format": "JSON",
        "AccessKeyId": AK,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureVersion": "1.0",
        "SignatureNonce": uuid.uuid4().hex,
        "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "RegionId": REGION,
    }
    p.update({k: v for k, v in params.items() if v is not None})
    qs = "&".join(f"{pct(k)}={pct(p[k])}" for k in sorted(p))
    sts = "POST&" + pct("/") + "&" + pct(qs)
    sig = base64.b64encode(hmac.new((SK + "&").encode(), sts.encode(), hashlib.sha1).digest()).decode()
    body = (qs + "&Signature=" + pct(sig)).encode()
    req = urllib.request.Request(HOST, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("API ERROR", e.code, e.read().decode()[:800])
        sys.exit(2)


def main():
    inst = sys.argv[1]
    cmd = sys.argv[2]
    if cmd == "--b64":
        cmd = base64.b64decode(sys.argv[3]).decode()
    r = call("RunCommand", Type="RunShellScript", CommandContent=base64.b64encode(cmd.encode()).decode(),
             ContentEncoding="Base64", **{"InstanceId.1": inst}, Timeout=600)
    inv = r.get("InvokeId")
    if not inv:
        print("NO InvokeId:", json.dumps(r)[:500]); sys.exit(2)
    print("InvokeId:", inv)
    deadline = time.time() + 560
    while time.time() < deadline:
        time.sleep(5)
        d = call("DescribeInvocationResults", InvokeId=inv, InstanceId=inst)
        res = (d.get("Invocation", {}).get("InvocationResults", {}) or {}).get("InvocationResult", [])
        if not res:
            continue
        it = res[0]
        st = it.get("InvocationStatus")
        if st in ("Success", "Failed", "Error", "Stopped", "Timeout"):
            out = it.get("Output", "")
            try:
                out = base64.b64decode(out).decode(errors="replace")
            except Exception:
                pass
            print(f"--- status={st} exit={it.get('ExitCode')} ---")
            print(out)
            sys.exit(0 if st == "Success" else 1)
    print("TIMEOUT waiting for result")
    sys.exit(3)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-shot ECS instance release via the ECS RPC API (signature v1, stdlib only).

Env: ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, ACS_REGION (default cn-shanghai)
Args: <instance_id> <expected_public_ip> <marker>

Flow (fail-closed):
  1. DescribeInstances pre-check: the target must be returned with an exact
     InstanceId match and the expected public IP, otherwise STOP (no delete).
  2. DeleteInstance with Force=true.
  3. Poll DescribeInstances until the instance is absent (verified release)
     or the poll budget runs out; report the last seen status.
     A poll only counts when it returns HTTP 200 with a valid Instances
     structure. Failed polls are retried until the budget runs out; budget
     expiry without a conclusive poll reports UNVERIFIED and exits nonzero
     (REVIEW_20261112 R15: an error response parses to an empty instance
     list and must never be read as "instance gone").
Every API response's HTTP status / RequestId / Code is printed verbatim.
"""
import base64, hashlib, hmac, json, os, sys, time, urllib.error, urllib.parse, urllib.request, uuid

AK = os.environ["ALIYUN_ACCESS_KEY_ID"]
SK = os.environ["ALIYUN_ACCESS_KEY_SECRET"]
REGION = os.environ.get("ACS_REGION", "cn-shanghai")
HOST = f"https://ecs.{REGION}.aliyuncs.com/"

INST = sys.argv[1]
EXPECTED_IP = sys.argv[2]
MARKER = sys.argv[3]


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
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"RawBody": raw[:800]}


def describe():
    code, r = call("DescribeInstances", InstanceIds=json.dumps([INST]), PageSize=10)
    insts = []
    if isinstance(r, dict):
        insts = (r.get("Instances") or {}).get("Instance") or []
    return code, r, insts


def main():
    print(f"=== {MARKER} BEGIN instance={INST} region={REGION} expected_ip={EXPECTED_IP} ===", flush=True)
    code, r, insts = describe()
    print(f"PRECHECK_HTTP={code} REQUESTID={r.get('RequestId')} TOTAL={r.get('TotalCount')}", flush=True)
    if code != 200:
        print("PRECHECK_RESPONSE=" + json.dumps(r)[:800], flush=True)
        print(f"=== {MARKER} RESULT=PRECHECK_ERROR ===", flush=True)
        return
    match = None
    for it in insts:
        if it.get("InstanceId") == INST:
            match = it
    if not match:
        print("PRECHECK_MATCH=NO (instance not returned by DescribeInstances)")
        print(f"=== {MARKER} RESULT=ABORT_NO_MATCH ===", flush=True)
        return
    ips = list((match.get("PublicIpAddress") or {}).get("IpAddress") or [])
    eip = (match.get("EipAddress") or {}).get("IpAddress")
    if eip:
        ips.append(eip)
    print(f"FOUND name={match.get('InstanceName')} status={match.get('Status')} ips={ips} charge={match.get('InstanceChargeType')} spot={match.get('SpotStrategy')}", flush=True)
    if EXPECTED_IP not in ips:
        print(f"PRECHECK_MATCH=NO (public IP mismatch: expected {EXPECTED_IP}, got {ips})")
        print(f"=== {MARKER} RESULT=ABORT_IP_MISMATCH ===", flush=True)
        return
    print("PRECHECK_MATCH=YES", flush=True)

    code, r = call("DeleteInstance", InstanceId=INST, Force="true")
    print(f"DELETE_HTTP={code} REQUESTID={r.get('RequestId')} CODE={r.get('Code')} MESSAGE={r.get('Message')}", flush=True)
    print("DELETE_RESPONSE=" + json.dumps(r)[:600], flush=True)
    if code != 200:
        print(f"=== {MARKER} RESULT=DELETE_ERROR ===", flush=True)
        return

    deadline = time.time() + 240
    last_status = None
    last_query_ok = False
    query_failures = 0
    while True:
        time.sleep(10)
        try:
            code, r, insts = describe()
        except Exception as e:
            code, r, insts = None, {"Exception": repr(e)}, []
        # REVIEW_20261112 R15: a verify poll is conclusive only on HTTP 200
        # with a valid Instances structure. An error response (503/403/...)
        # parses to an empty instance list; treating that as "absent" would
        # falsely report RELEASED_VERIFIED, so failed polls only retry.
        query_ok = code == 200 and isinstance(r, dict) and isinstance(r.get("Instances"), dict)
        if not query_ok:
            query_failures += 1
            last_query_ok = False
            print(f"VERIFY_HTTP={code} QUERY_OK=NO failures={query_failures} (retrying until deadline)", flush=True)
        else:
            last_query_ok = True
            present = any(it.get("InstanceId") == INST for it in insts)
            if present:
                last_status = [it.get("Status") for it in insts if it.get("InstanceId") == INST][0]
            print(f"VERIFY_HTTP={code} PRESENT={present} STATUS={last_status}", flush=True)
            if not present:
                print(f"=== {MARKER} RESULT=RELEASED_VERIFIED ===", flush=True)
                return
        if time.time() > deadline:
            if last_query_ok:
                print(f"=== {MARKER} RESULT=DELETE_ACCEPTED_STILL_PRESENT last_status={last_status} ===", flush=True)
                return
            print(f"=== {MARKER} RESULT=UNVERIFIED_QUERY_FAILED failures={query_failures} ===", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()

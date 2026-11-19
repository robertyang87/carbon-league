"""Sample Linux CPU quota and original match workers without attaching to or changing them."""
import argparse
import datetime
import gzip
import json
import os
from pathlib import Path
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cpu_status(path):
    values = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line)
    return values["Cpus_allowed_list"].strip()


def ticks(path):
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


def worker_state(pid):
    rows = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            # stat field 4 is PPID. Some managed kernels omit task/<tid>/children.
            if int((path / "stat").read_text().rsplit(")", 1)[1].split()[1]) != pid:
                continue
            command = (path / "cmdline").read_bytes().split(b"\x00")
            if b"--multiprocessing-fork" not in command:
                continue
            flags = {}
            env = (path / "environ").read_bytes().split(b"\x00")
            for field in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                prefix = field.encode() + b"="
                flags[field] = next((v[len(prefix):].decode() for v in env if v.startswith(prefix)), None)
            threads = [{"tid":int(t.name), "cpus":cpu_status(t), "ticks":ticks(t)} for t in (path/"task").iterdir()]
            rows.append(dict(pid=int(path.name), cpus=cpu_status(path), cpu_ticks=ticks(path), thread_limits=flags,
                             threads=threads))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--finished", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists() or args.finished.exists() or not Path(f"/proc/{args.pid}").is_dir():
        raise ValueError("monitor needs a running fresh observation window")
    cg = Path("/sys/fs/cgroup")
    now = time.monotonic()
    rows = []
    while True:
        cpu = {k:int(v) for k,v in (line.split() for line in (cg/"cpu.stat").read_text().splitlines())}
        rows.append(dict(utc=utc(), since_start=time.monotonic()-now, quota=(cg/"cpu.max").read_text().strip(),
                         cpus=(cg/"cpuset.cpus.effective").read_text().strip(), counters=cpu, workers=worker_state(args.pid)))
        finished = args.finished.exists()
        if finished or not Path(f"/proc/{args.pid}").is_dir():
            break
        time.sleep(args.interval)
    delta = {k:v-rows[0]["counters"][k] for k,v in rows[-1]["counters"].items()}
    doc = dict(utc_start=rows[0]["utc"], utc_end=rows[-1]["utc"], tick_hz=os.sysconf("SC_CLK_TCK"),
               batch_completed=finished, monotonic_seconds=rows[-1]["since_start"], cgroup_delta=delta,
               host_scope="cgroup aggregate for the recorded window; kernel counters also include other tasks",
               owner_pid=args.pid, samples=rows)
    tmp=args.out.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(doc, separators=(",",":"), allow_nan=False).encode(),mtime=0))
    os.replace(tmp,args.out)


if __name__ == "__main__":
    main()

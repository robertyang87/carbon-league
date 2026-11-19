"""Verify actual runtime imports, binary identities and cgroups in two offline containers, without mounting any policy."""
import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


def file_digest(path):
    with Path(path).open('rb') as fp:
        return hashlib.file_digest(fp, 'sha256').hexdigest()


def inside():
    import worker
    from importlib.metadata import version
    result=worker.probe()
    before=os.sched_getaffinity(0)
    # This is an import/capability check, not a benchmark. On the shared host,
    # use only an offline core for tensor operations after recording actual affinity.
    available=sorted(before & {2,3})
    if available:os.sched_setaffinity(0,{available[-1]})
    import numpy as np
    import torch
    from zerosum_env.envs.carbon.helpers import Board
    import zerosum_env.envs.carbon.helpers as helpers
    matrix=[[1.,2.],[3.,4.]]
    numpy_result=np.asarray(matrix) @ np.array([2.,3.])
    torch_result=torch.tensor(matrix) @ torch.tensor([2.,3.])
    if numpy_result.tolist()!=[8.,18.] or torch_result.tolist()!=[8.,18.] or Board.__name__!='Board':
        raise RuntimeError('runtime computation or helper import failed')
    if result['torch']!=torch.__version__ or result['numpy']!=np.__version__:
        raise RuntimeError('loaded binaries differ from version metadata')
    result['loaded']={'numpy':np.__version__,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
                      'helper':Board.__module__+'.'+Board.__name__,'requests':version('requests'),'jsonschema':version('jsonschema'),
                      'binary_smoke_result':torch_result.tolist(),
                      'sha256':{'torch_extension':file_digest(torch._C.__file__),
                                'libtorch_cpu':file_digest(Path(torch.__file__).parent/'lib/libtorch_cpu.so'),
                                'numpy_multiarray':file_digest(np.core._multiarray_umath.__file__),
                                'official_helpers':file_digest(helpers.__file__)}}
    if result['helper_sha256']!=result['loaded']['sha256']['official_helpers']:
        raise RuntimeError('helper bytes differ on import')
    print(json.dumps(result),flush=True)
    sys.stdin.readline()  # Keep both isolated processes alive for the Docker inspect readback.


@asynccontextmanager
async def container(image, mount_args, allowed):
    name='sapling-runtime-check-'+uuid.uuid4().hex[:10]
    args=['docker','run','--pull','never','--name',name,'-i','--network','none','--read-only',
          '--cpus','1.5','--memory','512m','--memory-swap','512m','--cpuset-cpus',allowed,
          '--tmpfs','/tmp:rw,nosuid,nodev,size=16m','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
          '-e','OMP_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1',*mount_args,image,'python','-u','/probe/runtime_probe.py','--inside']
    with tempfile.TemporaryFile() as log:
        process=await asyncio.create_subprocess_exec(*args,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=log)
        try:
            line=await asyncio.wait_for(process.stdout.readline(),90)
            if not line:
                log.seek(0);raise RuntimeError(log.read(2500).decode(errors='replace'))
            probe=json.loads(line)
            inspected=json.loads(subprocess.check_output(['docker','inspect',name],timeout=10))[0]
            c=inspected['HostConfig']; mounts=inspected['Mounts']
            if any(m.get('RW') for m in mounts if m['Type']=='bind'):
                raise RuntimeError('writable source mount')
            settings=dict(id=inspected['Id'],image=inspected['Image'],network=c['NetworkMode'],cpuset=c['CpusetCpus'],
                          nano_cpus=c['NanoCpus'],memory=c['Memory'],memory_swap=c['MemorySwap'],
                          readonly_root=c['ReadonlyRootfs'],binds=[{'target':m['Destination'],'readonly':not m['RW']} for m in mounts if m['Type']=='bind'])
            yield probe,settings
        finally:
            stop=await asyncio.create_subprocess_exec('docker','rm','-f',name,stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
            await stop.wait()
            if process.returncode is None:await asyncio.wait_for(process.wait(),10)


async def main(args):
    import live
    root=Path(__file__).resolve().parent
    if args.out.exists():raise FileExistsError(args.out)
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    mounts={'/probe/runtime_probe.py':Path(__file__).resolve(),'/probe/worker.py':root/'worker.py'}
    for i,site in enumerate(args.site):mounts[f'/runtime/site_{i}']=site.resolve(strict=True)
    if args.site:mounts['/runtime/official']=(live.ROOT/'code/env_src/carbon_challenge-master/zerosum_env').resolve()
    flags=[]
    for target,source in mounts.items():
        # Match the package directory name expected by Python's import machinery.
        dst='/runtime/official/zerosum_env' if target=='/runtime/official' else target
        flags.extend(['--mount',f'type=bind,src={source},dst={dst},readonly'])
    if args.site:flags.extend(['-e','PYTHONPATH='+':'.join(['/runtime/official',*[p for p in mounts if p.startswith('/runtime/site_')]])])
    async with AsyncExitStack() as stack:
        results=[await stack.enter_async_context(container(image,flags,args.cpus)) for _ in live.ROLES]
        probes,settings=[x[0] for x in results],[x[1] for x in results]
        blockers=live.eligibility(probes,settings)
        for p in probes:
            if p['loaded']['sha256']!=probes[0]['loaded']['sha256'] or p['loaded']['torch_cuda'] is not None:
                raise RuntimeError('runtime binary identities or CPU variant differ')
        outcome={'protocol_sha256':live.sha(root/'protocol.json'),'runner_sha256':file_digest(__file__),
                 'eligibility_source_sha256':live.sha(root/'live.py'),'worker_sha256':live.sha(root/'worker.py'),
                 'image_id':image,'image_reference':args.image,'mode':'read_only_installed_packages' if args.site else 'dependencies_baked_into_image',
                 'dependencies_mounted':bool(args.site),'strategy_source_mounted':False,'strategy_initializations':0,'action_calls':[0,0],
                 'docker_cgroup_driver':subprocess.check_output(['docker','info','--format','{{.CgroupDriver}}'],text=True).strip(),
                 'blockers':blockers,'containers':settings,'probes':probes,'counted':False,
                 'status':'BLOCKED_BEFORE_AGENT_LOAD' if blockers else 'RUNTIME_AND_RESOURCES_VERIFIED_NO_GAME'}
        args.out.parent.mkdir(exist_ok=True,parents=True);live.save(args.out,outcome)
        print(outcome['status'],outcome['mode'],blockers)


if __name__=='__main__':
    if sys.argv[1:]==['--inside']:
        inside()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--image',required=True);parser.add_argument('--site',action='append',type=Path,default=[])
        parser.add_argument('--cpus',default='2,3');parser.add_argument('--out',required=True,type=Path)
        asyncio.run(main(parser.parse_args()))

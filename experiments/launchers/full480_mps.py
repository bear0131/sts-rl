"""Run one training rank under a private MPS server with its calibrated plans."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def check_plans(directory):
    manifest = json.loads((directory / 'calibration.json').read_text())
    names = {f'wide-side{side}-plans-{a}.json' for side in (0, 1) for a in range(4)}
    names.update(f'block-upper-plans-{a}.json' for a in range(4))
    names.add('activated_plan_owner.so')
    if set(manifest) != names:
        raise ValueError('incomplete full480 calibration manifest')
    for name, sha in manifest.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'calibration checksum mismatch: {name}')


def start_cuda_client():
    driver = ctypes.CDLL('libcuda.so.1')
    context = ctypes.c_void_p()
    device = ctypes.c_int()
    sms = ctypes.c_int()

    def check(result):
        if result:
            raise RuntimeError(f'CUDA driver error {result}')

    check(driver.cuInit(0))
    check(driver.cuDeviceGet(ctypes.byref(device), 0))
    check(driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device))
    try:
        check(driver.cuCtxSetCurrent(context))
        check(driver.cuDeviceGetAttribute(ctypes.byref(sms), 16, device))
        check(driver.cuCtxSynchronize())
        if sms.value != 170:
            raise ValueError(f'full480 calibration requires 170 SMs, found {sms.value}')
    finally:
        check(driver.cuCtxSetCurrent(None))
        check(driver.cuDevicePrimaryCtxRelease_v2(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', required=True, type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('provide a training command after --')
    plans = args.plans.resolve()
    check_plans(plans)
    with tempfile.TemporaryDirectory(prefix='full480-mps-') as temp:
        pipe = Path(temp) / 'pipe'
        logs = Path(temp) / 'logs'
        pipe.mkdir()
        logs.mkdir()
        env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(pipe),
                   CUDA_MPS_LOG_DIRECTORY=str(logs), CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',
                   STS_FULL480_MPS_PLANS=str(plans))
        ctl = ['nvidia-cuda-mps-control']
        subprocess.run([*ctl, '-d'], env=env, check=True, timeout=30)
        try:
            subprocess.run([sys.executable, __file__, '--probe-cuda'], env=env, check=True, timeout=45)
            servers = subprocess.check_output(ctl, input='get_server_list\n', text=True, env=env, timeout=10).split()
            if len(servers) != 1 or not servers[0].isdigit():
                raise RuntimeError(f'expected a private MPS server, found {servers}')
            env['STS_FULL480_MPS_SERVER_PID'] = servers[0]
            return subprocess.run(command, env=env).returncode
        finally:
            subprocess.run(ctl, input='quit\n', text=True, env=env, check=True, timeout=30)


if __name__ == '__main__':
    if sys.argv[1:] == ['--probe-cuda']:
        start_cuda_client()
    else:
        sys.exit(main())

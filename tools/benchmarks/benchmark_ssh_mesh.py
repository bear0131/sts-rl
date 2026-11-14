"""Measure directed SSH file transfers without overlapping host assignments."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def pair_rounds(size: int) -> list[list[tuple[int, int]]]:
    """Each unordered pair occurs once; a host appears at most once per round."""
    if size < 2:
        raise ValueError("at least two hosts are required")
    players = list(range(size)) + ([None] if size % 2 else [])
    rounds = []
    for _ in range(len(players) - 1):
        rounds.append([(a, b) for a, b in zip(players[:len(players)//2],
                       reversed(players[len(players)//2:]), strict=True)
                       if a is not None and b is not None])
        players = [players[0], players[-1], *players[1:-1]]
    return rounds


REMOTE = r'''
from pathlib import Path
import hashlib,json,os,random,resource,shlex,signal,subprocess,time
folder=Path(config['directory']);folder.mkdir(parents=True,exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(4*1024*1024):h.update(block)
 return h.hexdigest()
def snapshot():
 result={}
 for key in ('rx_errors','rx_dropped','tx_errors','tx_dropped'):
  p=Path('/sys/class/net/eth0/statistics')/key
  result[key]=int(p.read_text()) if p.exists() else None
 return result
source=folder/'payload.bin'
if config['operation']=='prepare':
 assert not source.exists(),'refusing to overwrite an existing payload'
 rng=random.Random(config['seed'])
 with source.open('xb') as f:
  for _ in range(config['mib']):f.write(rng.randbytes(1024*1024))
  f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(bytes=source.stat().st_size,sha256=sha(source),seed=config['seed'])))
else:
 peer=config['destination'];destination=folder/('from-'+config['source_name']+'.bin')
 ssh=['ssh','-i',config['remote_identity'],'-o','BatchMode=yes','-o','Compression=no',
      '-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
      '-o','UserKnownHostsFile='+config['remote_known_hosts'],'-p',str(peer['port'])]
 endpoint=peer.get('user','root')+'@'+peer['host']
 argv=['rsync','--whole-file','--ignore-times','--no-compress','-e',shlex.join(ssh),
       str(source),endpoint+':'+shlex.quote(str(destination))]
 before=snapshot();usage=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
 proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 try:
  stdout,stderr=proc.communicate(timeout=config['timeout']);rc=proc.returncode
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM)
  try:stdout,stderr=proc.communicate(timeout=5)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
  rc=124
 elapsed=time.monotonic()-start;used=resource.getrusage(resource.RUSAGE_CHILDREN)
 result=dict(seconds=elapsed,bytes=source.stat().st_size,returncode=rc,
             verified=False,mib_per_sec=None,before=before,after=snapshot(),
             transfer_child_cpu_seconds=(used.ru_utime+used.ru_stime)-(usage.ru_utime+usage.ru_stime),
             child_max_rss_kib=used.ru_maxrss,gpu_util=None,gpu_note='CPU/network probe; GPU unused')
 if rc==0:
  checked=time.monotonic()
  verify=subprocess.run([*ssh,endpoint,shlex.join(['sha256sum',str(destination)])],capture_output=True,text=True,timeout=20)
  digest=verify.stdout.split()[0] if verify.returncode==0 and verify.stdout.split() else None
  result.update(sha256=digest,verified=digest==config['sha256'],verification_seconds=time.monotonic()-checked)
  if result['verified']:
   result['mib_per_sec']=result['bytes']/1024**2/elapsed
   # Only this invocation's exact received file is removed, after SHA validation.
   subprocess.run([*ssh,endpoint,shlex.join(['rm','--',str(destination)])],capture_output=True,timeout=15,check=True)
  else:result['error']='received SHA256 differs or verification failed: '+verify.stderr[-1000:]
 else:result['error']=stderr[-2000:]
 print(json.dumps(result))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    parser.add_argument('--remote-identity', required=True)
    parser.add_argument('--remote-known-hosts', required=True)
    parser.add_argument('--remote-root', default='/tmp/ssh-mesh-benchmark')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mib', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=45)
    parser.add_argument('--seed', type=int, default=20261115)
    args = parser.parse_args()
    hosts = json.loads(args.inventory.read_text(encoding='utf-8'))
    if not isinstance(hosts, list) or len(hosts) < 2:
        parser.error('inventory must contain at least two hosts')
    for h in hosts:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', h['name']):
            parser.error('host names must be unique simple labels')
        if not re.fullmatch(r'[A-Za-z0-9.-]+', h['host']) or not 1 <= int(h['port']) <= 65535:
            parser.error('invalid SSH endpoint')
    if len({h['name'] for h in hosts}) != len(hosts) or len({(h['host'], h['port']) for h in hosts}) != len(hosts):
        parser.error('duplicate host label or endpoint')
    if not (args.mib > 0 and args.workers > 0 and args.timeout > 0):
        parser.error('size, workers, and timeout must be positive')
    if args.output.exists():
        parser.error('output directory must be new to preserve earlier measurements')
    args.output.mkdir(parents=True)
    directory = args.remote_root.rstrip('/')+'/mesh-'+str(time.time_ns())
    started = time.time()
    report = dict(started_at=started, inventory=hosts, mib=args.mib, seed=args.seed,
                  workers=min(args.workers, len(hosts)//2), timeout=args.timeout,
                  directory=directory, rounds=pair_rounds(len(hosts)), results=[], preparations={},
                  method='single uncompressed SSH rsync; includes SSH startup and page-cache file I/O; excludes SHA validation; no shared hosts within a round')

    def remote(host, config, timeout):
        command = ['ssh', '-i', str(args.identity), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                   '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(args.known_hosts),
                   '-p', str(host['port']), host.get('user', 'root')+'@'+host['host'],
                   shlex.join([host.get('python', 'python3'), '-'])]
        config = dict(config, directory=directory)
        result = subprocess.run(command, input='config='+repr(config)+'\n'+REMOTE,
                                text=True, capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr[-2500:])
        return json.loads(result.stdout)

    def save():
        temp = args.output/'results.json.tmp'
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(args.output/'results.json')
        values = {(r['source'], r['destination']):r.get('mib_per_sec') for r in report['results']}
        with (args.output/'matrix-mib-per-sec.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f);writer.writerow(['source / destination', *[h['name'] for h in hosts]])
            for source in hosts:
                writer.writerow([source['name'], *[values.get((source['name'], dest['name']), '') for dest in hosts]])

    def prepare(index):
        return remote(hosts[index], dict(operation='prepare', mib=args.mib, seed=args.seed+index), 90)

    save()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(prepare, i):i for i in range(len(hosts))}
        for future in as_completed(futures):
            index = futures[future]
            try: report['preparations'][hosts[index]['name']] = future.result()
            except Exception as exc:
                report['preparations'][hosts[index]['name']] = dict(error=str(exc))
            save()
    if any('error' in p for p in report['preparations'].values()):
        report.update(completed=False, finished_at=time.time());save();return 1

    def measure(source, dest, round_index):
        row = dict(source=hosts[source]['name'], destination=hosts[dest]['name'], round=round_index, started_at=time.time())
        try:
            row.update(remote(hosts[source], dict(operation='transfer', destination=hosts[dest],
                       source_name=hosts[source]['name'], timeout=args.timeout,
                       remote_identity=args.remote_identity, remote_known_hosts=args.remote_known_hosts,
                       sha256=report['preparations'][hosts[source]['name']]['sha256']), args.timeout+60))
        except Exception as exc: row.update(verified=False, mib_per_sec=None, error=str(exc))
        row['finished_at'] = time.time()
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        for round_index, pairs in enumerate(report['rounds']):
            for reverse in (False, True):
                futures = [pool.submit(measure, b if reverse else a, a if reverse else b, round_index)
                           for a, b in pairs]
                for future in as_completed(futures):
                    row = future.result();report['results'].append(row);save()
                    print(json.dumps({k:row.get(k) for k in ('source','destination','mib_per_sec','verified','error')}), flush=True)
    expected = set(itertools.permutations([h['name'] for h in hosts], 2))
    report.update(completed=expected=={(r['source'],r['destination']) for r in report['results']}
                  and all(r['verified'] for r in report['results']), finished_at=time.time())
    save()
    print(json.dumps(dict(completed=report['completed'], directions=len(report['results']),
                          wall_seconds=time.time()-started, output=str(args.output))), flush=True)
    return 0 if report['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

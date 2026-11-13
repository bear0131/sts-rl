"""Bounded, read-only projection of the local worklog feed for the cloud Site."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.request


MAX_TEXT = 300_000
MAX_BYTES = 6 * 1024 * 1024
LOG_PATH = re.compile(r'worklog/\d{4}-\d{2}-\d{2}\.md\Z')


def public_text(text):
    """Remove credential-bearing syntax before truncation or remote persistence."""
    text = re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----',
                  '[私钥已隐藏]', str(text), flags=re.S)
    # SSH commands copied into user prompts can have an unlabelled password.
    text = re.sub(r'\b(?:ssh|sshpass)\s+[^\n]*', '[SSH 连接信息已隐藏]', text)
    text = re.sub(r'(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9_+/=.-]+',
                  '[授权信息已隐藏]', text)
    text = re.sub(r'(?i)((?:password|passwd|api[_-]?key|access[_-]?token|'
                  r'service[_-]?token|secret|private[_-]?key|send_token|密码|密钥)'
                  r'[\"\x27]?\s*[:=：]\s*)(?:\"[^\"]*\"|\x27[^\x27]*\x27|[^\s,;}]+)',
                  r'\1[已隐藏]', text)
    text = re.sub(r'(?i)(https?://)[^\s/@]+:[^\s/@]+@', r'\1[已隐藏]@', text)
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    return text


def public_feed(feed, *, source_id, label):
    """Deliberately omit local send tokens, paths/config and message controls."""
    entries = []
    for entry in feed.get('entries', [])[:3]:
        if not isinstance(entry, dict) or not LOG_PATH.fullmatch(entry.get('path', '')):
            continue
        text = public_text(entry.get('text', ''))
        entries.append(dict(path=entry['path'], text=text[-MAX_TEXT:],
                            truncated=bool(entry.get('truncated')) or len(text) > MAX_TEXT))
    commit = feed.get('commit', '')
    return dict(id=source_id, label=label, enabled=bool(feed.get('enabled')),
                branch=str(feed.get('branch', ''))[:160],
                commit=commit if re.fullmatch(r'[a-f0-9]{40,64}', commit or '') else None,
                updated_at=feed.get('updated_at'), attempted_at=feed.get('attempted_at'),
                error='来源同步失败，保留上次成功内容' if feed.get('error') else None,
                entries=entries)


def git_feed(config):
    def git(*args):
        result = subprocess.run(['git', '-C', config['repository'], *args],
            capture_output=True, timeout=30, env=dict(os.environ,
                GIT_TERMINAL_PROMPT='0', GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=8'))
        if result.returncode:
            raise RuntimeError('Git source unavailable')
        return result.stdout.decode('utf-8', 'replace')
    branch = config['branch']
    git('check-ref-format', 'refs/heads/' + branch)
    git('fetch', '--no-tags', 'origin',
        'refs/heads/' + branch + ':refs/remotes/origin/' + branch)
    commit = git('rev-parse', 'refs/remotes/origin/' + branch).strip()
    paths = sorted(p for p in git('ls-tree', '-r', '--name-only', commit, '--', 'worklog/').splitlines()
                   if LOG_PATH.fullmatch(p))[-3:]
    return dict(enabled=True, branch=branch, commit=commit, updated_at=time.time(), error=None,
                entries=[dict(path=p, text=git('show', commit + ':' + p)) for p in reversed(paths)])


def collect_worklog(config, previous=None):
    feeds = []
    old = {s['id']: s for s in (previous or {}).get('sources', [])}
    for source in config['worklog_sources']:
        identity, label = source['id'], source['label']
        try:
            if source['kind'] == 'local-feed':
                # This is the existing loopback-only log system, never an arbitrary URL.
                if not re.fullmatch(r'http://127\.0\.0\.1:[0-9]{1,5}/api/worklog', source['url']):
                    raise ValueError('Expected local worklog endpoint')
                with urllib.request.urlopen(source['url'], timeout=8) as response:
                    data = response.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError('Worklog feed too large')
                raw = json.loads(data)
            elif source['kind'] == 'git':
                raw = git_feed(source)
            else:
                raise ValueError('Unknown worklog source')
            feed = public_feed(raw, source_id=identity, label=label)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            feed = dict(old.get(identity) or public_feed({}, source_id=identity, label=label))
            feed.update(error='来源同步失败，保留上次成功内容', attempted_at=time.time())
        feeds.append(feed)
    return dict(schema_version=1, updated_at=time.time(), sources=feeds)


def publish_worklog(config, snapshot):
    body = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(body) > MAX_BYTES:
        raise ValueError('Worklog snapshot too large')
    # The fixed destination is a sibling of the monitor state, never a source path.
    target = config['remote_worklog_file']
    if not re.fullmatch(r'/root/autodl-tmp/[A-Za-z0-9_./-]+/state\.worklog\.json', target):
        raise ValueError('Invalid worklog destination')
    script = ('import pathlib,os,sys,hashlib\n'
              'p=pathlib.Path(' + repr(target) + ')\n'
              'data=sys.stdin.buffer.read(' + str(MAX_BYTES + 1) + ')\n'
              'assert len(data)<=' + str(MAX_BYTES) + '\n'
              'tmp=p.with_suffix(".upload.tmp")\n'
              'tmp.write_bytes(data)\nos.replace(tmp,p)\n'
              'print(hashlib.sha256(data).hexdigest())\n')
    args = ['ssh', '-i', config['ssh_key'], '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=8', '-o', 'UserKnownHostsFile=' + config['ssh_known_hosts'],
            '-p', str(config['port']), config['host'], 'python3 -c ' + shlex.quote(script)]
    result = subprocess.run(args, input=body, capture_output=True, timeout=30)
    digest = hashlib.sha256(body).hexdigest()
    if result.returncode or result.stdout.decode().strip() != digest:
        raise RuntimeError('Remote worklog copy unconfirmed')
    return dict(bytes=len(body), sha256=digest)

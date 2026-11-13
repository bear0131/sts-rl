import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from apps.training_site_sync import serve_state
from apps.training_worklog import collect_worklog, public_feed, public_text


def test_public_feed_keeps_log_content_but_never_local_write_authority():
    value = public_feed(dict(enabled=True, branch='codex/example', commit='a' * 40,
        updated_at=1, send_token='local-control-token', can_send=True,
        repository='private-path', entries=[dict(path='worklog/2026-11-14.md',
            text='### 验收\n196608行，16次Adam。'), dict(path='../secret', text='private')]),
        source_id='combat49', label='当前训练')
    assert value['entries'] == [dict(path='worklog/2026-11-14.md',
                                    text='### 验收\n196608行，16次Adam。', truncated=False)]
    assert value['commit'] == 'a' * 40
    assert all(x not in json.dumps(value) for x in ('local-control-token', 'can_send', 'private-path'))


def test_redaction_precedes_tail_and_covers_prompts_and_traceback_credentials():
    sample = ('before\nssh -p 123 root@host.example plain-password\n'
              'password="quoted secret" api_key=api-value\n'
              'Authorization: Bearer private-credential\n'
              '-----BEGIN RSA PRIVATE KEY-----\nprivate-lines\n-----END RSA PRIVATE KEY-----\n'
              'https://user:pass@example.com/path\nafter')
    value = public_text(sample)
    for secret in ('plain-password', 'quoted secret', 'api-value', 'private-credential',
                   'private-lines', 'user:pass'):
        assert secret not in value
    assert value.startswith('before\n') and value.endswith('\nafter')


def test_source_failure_retains_last_success_timestamp_and_text():
    previous = dict(sources=[dict(id='c49', label='current', updated_at=123,
                                  entries=[dict(path='worklog/2026-11-14.md', text='retained')])])
    config = dict(worklog_sources=[dict(id='c49', label='current', kind='git')])
    with patch('apps.training_worklog.git_feed', side_effect=RuntimeError('private detail')):
        result = collect_worklog(config, previous)
    assert result['sources'][0]['updated_at'] == 123
    assert result['sources'][0]['entries'] == previous['sources'][0]['entries']
    assert result['sources'][0]['error']
    assert 'private detail' not in json.dumps(result)


def test_fixed_ssh_request_cannot_select_files_or_modify_metrics():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'state.json'
        path.write_text('{"run_id":"current","rows":[],"comparisons":[]}')
        before = path.read_bytes()
        snapshot = dict(schema_version=1, updated_at=1, sources=[])
        path.with_suffix('.worklog.json').write_text(json.dumps(snapshot))
        assert json.loads(serve_state(path, 'worklog')) == snapshot
        for request in ('worklog:../private', 'worklog;id', 'worklog\ncurrent'):
            try:
                serve_state(path, request)
            except ValueError:
                pass
            else:
                raise AssertionError('unsafe command accepted')
        assert path.read_bytes() == before

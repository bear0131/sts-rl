import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from apps.training_trajectories import TrajectoryFeed, collect, remote, write_json


def test_stale_or_failed_refresh_keeps_inventory_and_source_time(tmp_path):
    saved = dict(run_id='active', global_step=10, captured_at=100, updated_at=110,
                 rows=[dict(deck=[['Bash', 0], ['Bash', 1], ['Bash', 1]],
                            relics=['Calling Bell', 'Matryoshka'], seed='9223372036854775807')])
    write_json(tmp_path / 'state.json', saved)
    write_json(tmp_path / 'status.json', dict(run_id='active', error='offline'))
    feed = TrajectoryFeed(dict(directory=str(tmp_path)))
    with patch('apps.training_trajectories.time.time', return_value=1500):
        actual = json.loads(feed.snapshot('active'))
    assert actual['rows'] == saved['rows']
    assert actual['captured_at'] == 100 and actual['updated_at'] == 110
    assert actual['stale'] and actual['error'] == 'offline'
    switched = json.loads(feed.snapshot('different-run'))
    assert switched['rows'] == [] and switched['updated_at'] is None
    assert '切换' in switched['error']


def test_images_cannot_read_outside_thumbnail_directory(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    name = hashlib.sha256(b'image').hexdigest() + '.png'
    (assets / name).write_bytes(b'png')
    feed = TrajectoryFeed(dict(directory=str(tmp_path)))
    assert feed.asset(name) == b'png'
    for bad in ('../state.json', name + '?q=1', '%2e%2e/state.json', 'z' * 64 + '.png'):
        assert feed.asset(bad) is None


def test_checkpoint_ssh_uses_rank_hosts_and_verified_keys():
    config = dict(root='/repo with space', run_id='run', ports=[11, 22],
                  hosts=['root@one', 'root@two'], ssh_key='key', ssh_known_hosts='known')
    with patch('apps.training_trajectories.subprocess.run', return_value=SimpleNamespace(
            returncode=0, stdout=b'ok')) as run:
        assert remote(config, 1, 'print(rank)') == b'ok'
    args = run.call_args.args[0]
    assert args[-4:] == ['-p', '22', 'root@two', 'bash -se']
    assert 'StrictHostKeyChecking=yes' in args and 'UserKnownHostsFile=known' in args
    assert b'\r' not in run.call_args.kwargs['input']


def test_unchanged_common_round_does_not_download_or_replay(tmp_path):
    write_json(tmp_path / 'state.json', dict(run_id='run', global_step=10, rows=['kept']))
    config = dict(run_id='run', ports=[11, 22], trajectories=dict(directory=str(tmp_path)))
    def answer(config, rank, code):
        # rank 0 has finished the next round; only round 10 is complete on both.
        return json.dumps([dict(run_id='run', rank=rank, global_step=step)
                           for step in ([10, 20] if rank == 0 else [10])]).encode()
    with patch('apps.training_trajectories.remote', side_effect=answer) as read, \
            patch('apps.training_trajectories.replay_bundle') as replay:
        collect(config)
    assert read.call_count == 2
    replay.assert_not_called()
    assert json.loads((tmp_path / 'state.json').read_text())['rows'] == ['kept']


def test_no_common_checkpoint_cannot_publish_partial_or_mixed_round(tmp_path):
    write_json(tmp_path / 'state.json', dict(run_id='run', global_step=5, rows=['kept']))
    config = dict(run_id='run', ports=[11, 22], trajectories=dict(directory=str(tmp_path)))
    def answer(config, rank, code):
        return json.dumps([dict(run_id='run', rank=rank, global_step=10 + rank)]).encode()
    with patch('apps.training_trajectories.remote', side_effect=answer), \
            pytest.raises(RuntimeError, match='no common complete'):
        collect(config)
    assert json.loads((tmp_path / 'state.json').read_text())['global_step'] == 5


def test_asset_names_use_unique_canonical_alias_and_do_not_guess_unknown_ids(tmp_path):
    import zipfile
    from apps.trajectory_assets import build_catalog
    jar = tmp_path / 'game.jar'
    with zipfile.ZipFile(jar, 'w') as archive:
        archive.writestr('localization/zhs/cards.json', json.dumps({
            'PanicButton': {'NAME': '应急按钮'}, 'Blood for Blood': {'NAME': '以血还血'}}))
        archive.writestr('localization/zhs/relics.json', '{}')
    rows = [dict(deck=[['Panic Button', 0], ['Blood For Blood', 1], ['Unknown', 0]], relics=[])]
    result = build_catalog(jar, tmp_path / 'assets', rows)
    assert result['cards']['Panic Button']['name'] == '应急按钮'
    assert result['cards']['Blood For Blood']['name'] == '以血还血'
    assert result['cards']['Unknown'] == {'name': 'Unknown', 'image': None}

import contextlib
import io
import json
import zipfile
from copy import deepcopy
from unittest.mock import patch

from apps.training_budget import budget_rows, merge_budget_records
from apps.training_monitor import Monitor, REMOTE


def record(rank, counts, step=100, round_number=1):
    stats = {'common/budget_decisions': sum(counts),
             'common/budget_total_simulations': sum(b*c for b,c in zip((0,32,128,512,2048), counts))+20,
             'common/budget_audit_simulations': 20,
             'common/budget_total_transitions': 5000,
             'common/budget_audit_transitions': 500}
    stats.update({'common/budget_selected_'+str(b):c
                  for b,c in zip((0,32,128,512,2048),counts)})
    return dict(rank=rank, global_timesteps=step, rollout_round=round_number, stats=stats)


def test_counts_are_weighted_by_decisions_and_audit_is_separate():
    rows = budget_rows([record(0,[9,1,0,0,0]),record(1,[0,90,0,0,0])],2)
    row=rows[0]; t=row['tiers']['all']
    assert row['complete'] and row['valid']
    assert t['decisions']==100 and t['counts']==[9,91,0,0,0]
    assert t['simulations']/t['decisions']==29.12
    assert t['audit_simulations']==40
    assert t['audit_transitions']==1000 and t['transitions']==9000
    assert row['tiers']['boss']['decisions']==0


def test_missing_ranks_replacement_and_invalid_counters():
    r=record(0,[1,1,0,0,0])
    assert not budget_rows([r],2)[0]['complete']
    updated=record(0,[2,1,0,0,0])
    assert merge_budget_records([r],[updated,updated])==[updated]
    bad=deepcopy(r);bad['stats']['common/budget_decisions']=99
    assert not budget_rows([bad],1)[0]['valid']
    bad=deepcopy(r);bad['stats']['common/budget_total_transitions']=float('nan')
    assert not budget_rows([bad],1)[0]['valid']
    assert budget_rows([],4)==[]


def test_poll_restart_does_not_double_count_or_leak_to_new_run(tmp_path):
    config=dict(run_id='budget-run',iteration_steps=None,rollout_mode='complete_games',ports=[1,2])
    path=tmp_path/'state.json'
    monitor=Monitor(config,path,30)
    def read(rank):
        return dict(rank=rank, rows=[],budget_records=[record(rank,[1,2,3,4,5])])
    with patch.object(monitor,'read_rank',side_effect=read):
        monitor.poll()
    first=monitor.state['search_budget_rows']
    restarted=Monitor(config,path,30)
    with patch.object(restarted,'read_rank',side_effect=read):
        restarted.poll()
    assert restarted.state['search_budget_rows']==first
    assert len(restarted.state['search_budget_records'])==2
    assert not Monitor(dict(config,run_id='other'),path,30).state.get('search_budget_rows')


def test_remote_backfills_budget_cursor_and_keeps_partial_line(tmp_path):
    run=tmp_path/'runs'/'example';run.mkdir(parents=True)
    legacy=dict(global_timesteps=1,rollout_sec=1,rollout_round=1)
    measured=dict(global_timesteps=2,rollout_sec=1,rollout_round=2,
        small_playout_stats_rank=record(1,[1,0,0,0,0])['stats'])
    path=run/'rank1.log'
    data=(json.dumps(legacy)+'\n'+json.dumps(measured)+'\n').encode()
    path.write_bytes(data+b'{"incomplete"')
    def read(cursor):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(REMOTE,dict(ROOT=str(tmp_path),RUN='example',RANK=1,CURSOR=cursor))
        return json.loads(out.getvalue())
    # Existing log cursor cannot suppress the first budget backfill.
    result=read(dict(inode=path.stat().st_ino,offset=len(data)))
    assert result['rows']==[]
    assert result['budget_records']==[dict(record(1,[1,0,0,0,0],2,2))]
    assert result['cursor']['budget_offset']==len(data)
    assert read(result['cursor'])['budget_records']==[]
    path.write_text(json.dumps(measured)+'\n')
    assert len(read(result['cursor'])['budget_records'])==1


def test_shared_source_must_match_the_run_and_rank_on_the_controller(tmp_path):
    root = tmp_path/'controller'
    run = root/'runs'/'example'
    run.mkdir(parents=True)
    (run/'rank1.log').touch()
    (run/'rank1.pid').write_text('1234567')
    proc = tmp_path/'proc'
    cwd = proc/'cwd'
    cwd.mkdir(parents=True)
    (proc/'cmdline').write_bytes(b'python\0-m\0rl.training.distributed_sb3\0--run-id\0example\0')
    def read(plan):
        (root/'train-plan.json').write_text(json.dumps(plan))
        def path_for(value):
            return proc if value=='/proc/1234567' else type(tmp_path)(value)
        out = io.StringIO()
        with patch('pathlib.Path', side_effect=path_for), contextlib.redirect_stdout(out):
            exec(REMOTE, dict(ROOT=str(root),RUN='example',RANK=1,CURSOR={}))
        return json.loads(out.getvalue())['alive']
    plan = dict(rank=1,run=str(run),cwd=str(cwd))
    assert read(plan)
    assert not read(dict(plan,rank=0))
    assert not read(dict(plan,run=str(root/'runs'/'old')))
    assert not read(dict(plan,cwd=str(tmp_path/'elsewhere')))


def test_original_checkpoint_reads_only_present_card_limit_counts(tmp_path):
    run = tmp_path/'runs'/'example'
    bundles = run/'recovery'/'rank0'
    bundles.mkdir(parents=True)
    (run/'rank0.log').write_text('')
    for step, tiers in ((10, ['common', 'boss']), (20, ['common','elite','boss'])):
        bundle = bundles/f'step-{step:012d}'
        bundle.mkdir()
        (bundle/'manifest.json').write_text(json.dumps(dict(global_step=step,rank=0,run_id='example')))
        with zipfile.ZipFile(bundle/'model.zip','w') as z:
            z.writestr('data',json.dumps({'_small_playout_stats':{t+'/turn_card_limit_decisions':i for i,t in enumerate(tiers)}}))
    def read(cursor):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(REMOTE,dict(ROOT=str(tmp_path),RUN='example',RANK=0,CURSOR=cursor,ORIGINAL_PPO=True))
        return json.loads(stream.getvalue())
    first = read({})
    assert first['original_card_limits'] == [dict(step=20,count=3)]
    assert read(first['cursor'])['original_card_limits'] == []

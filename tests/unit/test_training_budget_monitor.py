import contextlib
import io
import json
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

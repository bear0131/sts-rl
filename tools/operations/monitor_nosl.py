"""Read a rank-0 JSON-lines training log and record isolated search interventions."""
import argparse
import json
from pathlib import Path
import time

from rl.training.monitoring import assess, RECOVERY_STEPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--record-change', help='Parameter name, only after verified save/stop/resume')
    parser.add_argument('--at-step', type=int)
    args = parser.parse_args()
    state = json.loads(args.state.read_text()) if args.state.exists() else {'interventions':[]}
    records = []
    for line in args.log.read_text(errors='replace').splitlines():
        if line.startswith('{'):
            try: records.append(json.loads(line))
            except json.JSONDecodeError: pass  # writer can be in the middle of its final line
    if args.record_change:
        index=len(state['interventions'])
        if index >= len(RECOVERY_STEPS) or args.record_change != RECOVERY_STEPS[index][0] or args.at_step is None:
            raise ValueError('intervention must be the next coordinate and have an explicit global step')
        state['interventions'].append(dict(parameter=args.record_change, value=RECOVERY_STEPS[index][1],
            global_step=args.at_step,time=time.time()))
    result = assess(records,state['interventions'],now=time.time())
    state.update(last_check=time.time(), result=result)
    args.state.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.state.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
    temporary.replace(args.state)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

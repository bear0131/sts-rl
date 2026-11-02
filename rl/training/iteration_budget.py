"""Fail closed on a hung/over-budget round; only prior common saves can resume."""
import json
import os
import threading
import time


class IterationBudget:
    def __init__(self, seconds, *, rank, directory):
        self.seconds = seconds
        self.rank = rank
        self.directory = directory
        self.timer = None

    def start(self, iteration):
        self.cancel()
        if not self.seconds:
            return
        started = time.monotonic()
        def expired():
            report = dict(reason='iteration-wall-budget', rank=self.rank,
                          iteration=iteration, limit=self.seconds,
                          elapsed=time.monotonic()-started, exit_code=124)
            # os._exit also interrupts a native/GPU wait. No partial learner or
            # censored game is published as a successful training round.
            try:
                if self.directory is not None:
                    self.directory.mkdir(parents=True, exist_ok=True)
                    (self.directory / f'round-timeout-rank{self.rank}.json').write_text(json.dumps(report))
                print(json.dumps(report), flush=True)
            finally:
                os._exit(124)
        self.timer = threading.Timer(self.seconds, expired)
        self.timer.daemon = True
        self.timer.start()

    def cancel(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

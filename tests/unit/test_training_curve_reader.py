from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from apps.training_monitor import REMOTE


class TrainingCurveReader(unittest.TestCase):
    def test_observer_measurements_and_cursor_share_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'runs' / 'trial'
            folder.mkdir(parents=True)
            (folder / 'rank1.log').touch()
            sample = dict(step=32, round=4, run_id='trial', fields={'ppo/update_cosine/all': .2})

            def directions(path, name, seen):
                self.assertEqual((path, name, seen), (folder / 'update-directions', 'trial', [3]))
                return [sample], [3, 4]

            out = StringIO()
            with redirect_stdout(out):
                exec(REMOTE, dict(ROOT=tmp, RUN='trial', RANK=1, ORIGINAL_PPO=True,
                                  CURSOR={'direction_rounds': [3]}, read_original_directions=directions))
            data = json.loads(out.getvalue())
            self.assertEqual(data['original_directions'], [sample])
            self.assertEqual(data['cursor']['direction_rounds'], [3, 4])

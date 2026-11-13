"""CPU root export with exact replay of device-incompatible callback queues.

Only portable battle states cross the device boundary. A short committed action
suffix recreates pending callbacks on CUDA. The live game is never replayed or
mutated; a restored private clone bootstraps an initially pending root.
"""
from .cuda_battle import load_artifacts
from .resident_chains import root_payload


class ResidentRootExporter:
    def __init__(self, envs, artifacts):
        self.envs = [e.unwrapped for e in envs]
        _, _, self.bridge = load_artifacts(artifacts)
        self.cached = [None]*len(envs)

    def _remember(self, env, previous, action):
        if env._phase() != 'combat' or env._at_combat_exit_anchor():
            return None
        state = self.bridge.snapshot(env.thick.bc)
        if state['portable']:
            previous = dict(snapshot=state, prefix=sum(1 << k for k in (env._multi or ())), replay=[])
        if previous is not None:
            previous['replay'].append(int(action)-env._policy_offset)
        return previous

    def before_step(self, actions):
        for i, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            self.cached[i] = self._remember(env, self.cached[i], action)

    def export(self, indices):
        result = []
        for i in indices:
            env = self.envs[i]
            root = root_payload(env, self.bridge)
            root['replay'] = []
            if not root['snapshot']['portable']:
                if self.cached[i] is None:
                    shadow = env.fork_run()
                    current = None
                    def remember(branch, record):
                        nonlocal current
                        action = (record['action'] if record['kind'] == 'thin' else
                                  {int(b): a for a, b in branch._joint_policy_actions()}[record['action_bits']])
                        current = self._remember(branch, current, action)
                    try:
                        shadow.restore_training_state(env.export_training_state(), replay_observer=remember)
                    finally:
                        shadow.close()
                    self.cached[i] = current
                if self.cached[i] is None:
                    # A battle can enter its very first choice with callbacks
                    # already queued (there is no earlier combat action to
                    # replay). Keep the exact native state in its owner worker;
                    # the learner routes this worker's paired group through the
                    # existing native-branch/GPU-inference service.
                    root['native_fallback'] = 'pending-without-portable-origin'
                else:
                    root.update(self.cached[i])
            # The following real step may extend our cached suffix after send.
            result.append(dict(root, replay=list(root['replay'])))
        return result

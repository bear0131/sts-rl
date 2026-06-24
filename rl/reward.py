from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class RewardConfig:
    floor: float = 1.0
    floor_base: float = 1.03  # each floor f pays floor * floor_base**f (depth premium)
    boss_rewards: tuple = (25.0, 45.0, 80.0)  # per-act boss clear (act1, act2, act3)
    death: float = 0.0
    hp_delta: float = 0.0


@dataclass(frozen=True)
class RewardState:
    hp: int
    act: int
    floor: int
    outcome: str   # "UNDECIDED" / "PLAYER_VICTORY" / "PLAYER_LOSS"


class RewardCalculator:
    def __init__(self, cfg=None):
        self.cfg = cfg or RewardConfig()

    def compute(self, prev: RewardState, now: RewardState, terminated: bool) -> float:
        cfg = self.cfg
        r = 0.0

        # Each newly reached floor f pays floor_base**f: an exponential depth
        # premium that offsets gamma's per-floor discount and then mildly
        # over-weights deep floors (1.03 * 0.999**~20steps ~= 1.009 net/floor).
        for f in range(prev.floor + 1, now.floor + 1):
            r += cfg.floor * (cfg.floor_base ** f)
        r += cfg.hp_delta * (now.hp - prev.hp)

        if now.outcome == "PLAYER_LOSS":
            r += cfg.death

        # Crossing an act boundary (act N -> N+1) means we just cleared act N's
        # boss; award boss_rewards[N-1] for each boundary crossed this step.
        for act in range(prev.act, now.act):
            if 1 <= act <= len(cfg.boss_rewards):
                r += cfg.boss_rewards[act - 1]

        return float(r)


def _gc_state(gc) -> RewardState:
    return RewardState(
        hp=gc.cur_hp,
        act=gc.act,
        floor=gc.floor_num,
        outcome=gc.outcome.name,
    )


class ShapedRewardWrapper(gym.Wrapper):
    """Replace env reward with shaped reward from RewardCalculator.

    Wraps SlayTheSpireEnv (or any env whose .thick.gc exposes cur_hp/act/
    floor_num/outcome). Stores the raw env reward in info["env_reward"].
    """

    def __init__(self, env, cfg=None):
        super().__init__(env)
        self._calc = RewardCalculator(cfg)
        self._prev: RewardState | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev = _gc_state(self.env.thick.gc)
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        now = _gc_state(self.env.thick.gc)
        shaped = self._calc.compute(self._prev, now, terminated)
        self._prev = now
        info["env_reward"] = env_reward
        return obs, shaped, terminated, truncated, info

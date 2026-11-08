from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class RewardConfig:
    floor: float = 0.25
    boss_rewards: tuple[float, ...] = (
        50.0, 100.0, 200.0,
    )  # act1, act2, act3 boss clear
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

        # Each newly reached floor pays a small flat reward. Boss clears are
        # rewarded separately below and intentionally dominate floor progress.
        r += cfg.floor * max(0, now.floor - prev.floor)
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


class StrategicRewardWrapper(ShapedRewardWrapper):
    """The current upper's reward, now credited to every actual action."""

    def step(self, action):
        from env.core.outer_return import strategic_reward
        obs, raw, terminated, truncated, info = self.env.step(action)
        now = _gc_state(self.env.thick.gc)
        reward = strategic_reward(self._prev.act, self._prev.floor,
            dict(act=now.act, floor=now.floor, outcome=now.outcome),
            done=terminated, censored=bool(truncated or info.get('dead_screen')))
        self._prev = now
        info['env_reward'] = raw
        return obs, reward, terminated, truncated, info


class CombatGauntletRewardWrapper(gym.Wrapper):
    """Reward for the combat-gauntlet pretraining (doc/night_0625/
    combat_pretrain_decision.md): +`win` each time a fight is cleared. An optional
    end-of-combat penalty subtracts ``hp_loss_coef * net HP lost`` for both wins
    and the terminal loss. With the default coefficient 0, the original reward is
    unchanged and HP value is learned only from the future wins death cuts off.

    Wraps a SlayTheSpireEnv whose backend is a SimCombatBackend; it reads the
    backend's cumulative `combats_won` counter and pays for each new clear. Mirrors
    ShapedRewardWrapper: the env's own reward is ignored and stashed in
    info["env_reward"].
    """

    def __init__(self, env, win: float = 1.0, hp_loss_coef: float = 0.0):
        super().__init__(env)
        if hp_loss_coef < 0:
            raise ValueError("hp_loss_coef must be non-negative")
        self.win = win
        self.hp_loss_coef = hp_loss_coef
        self._prev_won = 0
        self._combat_start_hp = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        backend = self.env.unwrapped.thick
        self._prev_won = backend.combats_won
        self._combat_start_hp = backend.combat_state()[0]
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        backend = self.env.unwrapped.thick
        won_now = backend.combats_won
        r = self.win * (won_now - self._prev_won)
        combat_finished = won_now > self._prev_won or terminated
        if combat_finished:
            combat_end_hp = backend.combat_state()[0]
            hp_lost = max(0, self._combat_start_hp - combat_end_hp)
            hp_penalty = self.hp_loss_coef * hp_lost
            r -= hp_penalty
            info["combat_hp_lost"] = hp_lost
            info["combat_hp_penalty"] = hp_penalty
            self._combat_start_hp = combat_end_hp
        self._prev_won = won_now
        if terminated or truncated:
            info["combats_won"] = won_now
            # the episode's act (= act of the fixed level L), for per-act metrics.
            info["combat_act"] = self.env.unwrapped.thick.gc.act
        info["env_reward"] = env_reward
        return obs, r, terminated, truncated, info

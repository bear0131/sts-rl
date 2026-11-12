"""Checkpointed preparation gates for deck-latent PPO and search teaching."""
from copy import deepcopy
from dataclasses import asdict, dataclass

from rl.algorithms.search_auxiliary import SearchTrainingConfig
from rl.evaluation.anchor_readiness import readiness_metrics


@dataclass(frozen=True)
class DeckTrainingSchedule:
    first_check: int = 100
    check_interval: int = 20
    consecutive_checks: int = 3
    anchor_rounds: int = 20
    ramp_rounds: int = 20
    validation_games: int = 64
    validation_max_steps: int = 20000
    validation_seconds: float = 900.
    validation_seed: int = 27182818
    minimum_natural: int = 32
    minimum_anchors: int = 64
    minimum_ev: float = .05
    search_qualified: bool = False
    search_backend: str = 'native-cpu'
    search_cuda_artifacts: str = ''
    search_cuda_wave_capacity: int = 4096
    search_inference_cache_bytes: int = 0
    anchor_start_round: int | None = None

    def __post_init__(self):
        SearchTrainingConfig(backend=self.search_backend, cuda_artifacts=self.search_cuda_artifacts,
            cuda_wave_capacity=self.search_cuda_wave_capacity, inference_cache_bytes=self.search_inference_cache_bytes)
        if min(self.check_interval, self.consecutive_checks, self.anchor_rounds, self.ramp_rounds,
               self.validation_games, self.validation_max_steps, self.minimum_natural, self.minimum_anchors) < 1:
            raise ValueError('deck training schedule requires positive bounded counts')
        if self.first_check < 0 or not 30 <= self.validation_seconds <= 3600:
            raise ValueError('invalid deck readiness validation budget')
        if self.minimum_natural > self.validation_games:
            raise ValueError('readiness cannot require more natural games than its budget')
        if self.anchor_start_round is not None and (type(self.anchor_start_round) is not int
                                                  or self.anchor_start_round < 0):
            raise ValueError('explicit anchor start must be a non-negative completed round')


def configure(model, config=None):
    config = config if config is not None else getattr(model, 'deck_training_schedule', None)
    if config is None:
        return None
    if getattr(model.policy.core, 'model_contract', None) != 'deck-latent-combat-v1':
        raise ValueError('deck training schedule requires its own model geometry')
    schedule = DeckTrainingSchedule(**config)
    model.deck_training_schedule = asdict(schedule)
    if not hasattr(model, 'deck_training_state'):
        if model.policy.combat_value_target != 'whole-run' or getattr(model, 'deck_search_config', None):
            raise ValueError('a new staged run must begin with whole-run PPO warmup')
        model.deck_training_state = dict(contract='deck-latent-stages-v1', stage='warmup',
            stage_start=0, good_checks=0, last_check=-1, previous_floor=None, validation=None)
    if model.deck_training_state['contract'] != 'deck-latent-stages-v1':
        raise ValueError('unknown deck training stage checkpoint contract')
    return schedule


def check_due(state, schedule, completed_rounds):
    return (completed_rounds >= schedule.first_check
            and completed_rounds % schedule.check_interval == 0
            and completed_rounds != state['last_check'])


def advance(state, schedule, completed_rounds, moments):
    state = deepcopy(state)
    if not check_due(state, schedule, completed_rounds):
        raise ValueError('stage validation must occur at its scheduled boundary')
    metrics = readiness_metrics(moments)
    natural = moments['natural']
    floor = moments['floor_sum'] / natural if natural else 0.
    healthy = (natural >= schedule.minimum_natural and moments['anchors'] >= schedule.minimum_anchors
        and moments['timeout'] == 0 and moments.get('failed', 0) == 0
        and moments['censored'] <= .1 * schedule.validation_games
        and (moments['relic_offered'] < 10 or moments['relic_taken'] / moments['relic_offered'] >= .01)
        and (state['previous_floor'] is None or floor >= .9 * state['previous_floor']))
    passed = healthy and metrics['explained_variance'] is not None and metrics['explained_variance'] > schedule.minimum_ev
    state.update(last_check=completed_rounds, previous_floor=floor,
        validation=dict(moments=moments, **metrics, passed=bool(passed)))
    state['good_checks'] = state['good_checks'] + 1 if passed else 0
    if state['stage'] == 'warmup' and state['good_checks'] >= schedule.consecutive_checks:
        state.update(stage='anchor', stage_start=completed_rounds)
    elif (state['stage'] == 'anchor' and passed and schedule.search_qualified
          and completed_rounds - state['stage_start'] >= schedule.anchor_rounds):
        state.update(stage='search', stage_start=completed_rounds)
    return state


def apply_stage(model, schedule, completed_rounds, *, rank=0):
    state = model.deck_training_state
    if (state['stage'] == 'warmup' and schedule.anchor_start_round is not None
            and completed_rounds >= schedule.anchor_start_round):
        # An explicit operator decision is distinct from passed validation.
        # Do not invent good_checks, replace its evidence, or reset on resume.
        state.update(stage='anchor', stage_start=completed_rounds,
                     transition_reason='configured-anchor-start')
    target = 'whole-run' if state['stage'] == 'warmup' else 'pre-reward-anchor'
    model.policy.combat_value_target = target
    model.policy_kwargs['combat_value_target'] = target
    model.deck_search_config = None
    if state['stage'] == 'search':
        if not schedule.search_qualified:
            raise ValueError('a search-stage checkpoint cannot resume on an unqualified backend')
        progress = min(1., max(0., (completed_rounds - state['stage_start']) / schedule.ramp_rounds))
        fraction = .25 + .5 * progress if progress <= .5 else progress
        model.deck_search_config = asdict(SearchTrainingConfig(fraction=fraction, coefficient=.1 * progress,
            target_std=state['validation']['target_std'], seed=20261111 + 1_000_003 * rank,
            backend=schedule.search_backend, cuda_artifacts=schedule.search_cuda_artifacts,
            cuda_wave_capacity=schedule.search_cuda_wave_capacity,
            inference_cache_bytes=schedule.search_inference_cache_bytes))
    model.training_objective_contract = ('deck-latent-ppo-search-v1' if state['stage'] == 'search'
                                         else 'deck-latent-ppo-' + state['stage'] + '-v1')

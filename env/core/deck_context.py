"""Public context extension for the cached-deck policy, without reward changes."""
import numpy as np
from gymnasium import spaces

from . import encode
from .outer_return import permanent_misc


def extend_space(space):
    space.spaces.update(
        deck_latent_context=spaces.Box(0, 1, (1,), np.float32),
        outer_deck_misc=spaces.Box(0, np.inf, (encode.DECK_SLOTS,), np.float32),
        pile_upgraded_counts=spaces.Box(0, np.inf, (3, encode.N_CARDS), np.float32),
        pile_upgrade_sums=spaces.Box(0, np.inf, (3, encode.N_CARDS), np.float32))


def project(env, observation, canonical):
    """Reference full projection; no hidden pile order or future rewards.

    Counts include all vocabulary IDs and exhaust, not only the older rule
    language's subset. Native incremental consumers must match this reference.
    """
    counts = np.zeros((3, encode.N_CARDS), np.float32)
    upgrades = np.zeros_like(counts)
    for zone, pile in enumerate(('draw', 'discard', 'exhaust')):
        for card in canonical.get(pile, ()):
            card_id, up = encode._card_id_idx(card['id']), card.get('up', 0)
            counts[zone, card_id] += up > 0
            upgrades[zone, card_id] += up / encode.UPGRADE_SCALE
    _publish(env, observation, counts, upgrades)


def _publish(env, observation, counts, upgrades):
    def assign(key, value):
        if key in observation:
            observation[key][...] = value
        else:
            observation[key] = value
    assign('deck_latent_context', np.ones(1, np.float32))
    assign('outer_deck_misc', permanent_misc(env.thick.gc, env.thick.bc))
    assign('pile_upgraded_counts', counts)
    assign('pile_upgrade_sums', upgrades)
    if env._phase() == 'combat':
        # Read-only numeric projection of the current public map. Requesting
        # its ordinary map view neither advances the run nor reveals rewards.
        view = env.thick.encoding_map_observation(dict(phase='map'))
        encode._encode_options(observation, dict(phase='map', options={}), _map_observation=view)


def project_native_root(env, observation, packed=None):
    """Supplement the packed root using the retained extension's live pile data."""
    if packed is not None:
        _publish(env, observation, packed[0], packed[1])
        return
    from schemas.combat_state import _card, _sim_card_id
    raw = env.thick.bc.to_dict()
    piles = {pile: [_card(_sim_card_id(name), upgrades) for name, upgrades in raw[pile]]
             for pile in ('draw', 'discard', 'exhaust')}
    project(env, observation, piles)

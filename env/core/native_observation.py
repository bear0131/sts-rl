"""Encode native combat branches using the shared observation/action codecs.

Execution and observation contracts are independent: a native-v1 search may
encode a branch for an AR checkpoint by supplying its contract and policy IDs.
Persistent run metadata is explicit; branch HP and potions override that root.
"""
from . import _BUILD  # noqa: F401 (native/schema bootstrap)
from . import action_space as A, encode, native_actions as C
from schemas.combat_state import combat_select_options, normalize_sim


def encode_native_observation(raw_state, *, root_global, combat_action_contract,
                              combat_only=False, policy_ids=None,
                              card_language=False, combat_language=False,
                              global_map_view=False, deck_size_observation=False,
                              neow_observation=False, event_option_observation=False,
                              interactive_reasoning=False):
    """Encode without an environment, model, or changes to either input.

    ``selection_prefix`` in raw_state describes an AR multi-select prefix;
    ``policy_ids`` must describe that same branch/prefix. If omitted, use the
    native transport's ``joint_action_ids`` (including its empty-mask default).
    The caller owns legality queries and the root metadata snapshot. This
    function neither samples futures nor advances a projected boundary.
    """
    if combat_action_contract not in ('legacy', 'native-v1', *C.NEURAL_CONTRACTS):
        raise ValueError(f'unsupported combat action contract: {combat_action_contract}')
    canon = normalize_sim(raw_state)
    canon['phase'] = 'combat'
    canon['arg_step'] = A.ROOT
    if int(raw_state.get('input_state', 1)) == 2:
        select = raw_state.get('select') or {}
        if select:
            canon['options'] = {'select': combat_select_options(select)}
    if combat_action_contract == C.AUTOREGRESSIVE_CONTRACT and 'selection_prefix' in raw_state:
        prefix = int(raw_state['selection_prefix'])
        sel = canon.get('options', {}).get('select')
        if sel is not None:
            sel['hand_selected'] = [i for i in range(10) if prefix >> i & 1]
            sel['pick_remaining'] = max(
                0, int(sel.get('pick_remaining', sel.get('pick_max', 0))) - prefix.bit_count())
    global_state = dict(root_global)
    global_state['hp'] = int(raw_state['player']['hp'])
    global_state['max_hp'] = int(raw_state['player']['max_hp'])
    if 'potions' in raw_state:
        global_state['potions'] = list(raw_state['potions'])
    canon['global'] = global_state
    if combat_only:
        obs = encode.encode_combat(canon, card_language=card_language,
                                   combat_language=combat_language)
    else:
        obs = encode.encode(canon, global_map_view=global_map_view,
                            card_language=card_language, combat_language=combat_language,
                            deck_size_observation=deck_size_observation,
                            neow_observation=neow_observation,
                            event_option_observation=event_option_observation,
                            interactive_reasoning=interactive_reasoning)
    if combat_action_contract in C.NEURAL_CONTRACTS:
        if policy_ids is None:
            policy_ids = raw_state.get('joint_action_ids', ())
        obs['combat_legal'] = C.joint_mask(policy_ids, contract=combat_action_contract)[
            C.policy_offset(combat_action_contract):]
    return obs

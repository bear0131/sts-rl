"""Native bit handles retain the backend's source and target identities."""
from types import SimpleNamespace

from env.core.native_actions import action_bits, complete_actions


def test_native_potions_keep_noncontiguous_slots_and_target_ids():
    battle=SimpleNamespace(rule_actions=lambda:[],legal_potions=lambda:[
        {'slot':2,'requires_target':True,'targets':[1,4]},
        {'slot':4,'requires_target':False,'targets':[]}])
    assert complete_actions(battle)==(1<<29 | 1<<16 | 2,1<<29 | 4<<16 | 2,1<<29 | 4)


def test_native_single_selection_keeps_full_index():
    battle=SimpleNamespace(rule_actions=lambda:[SimpleNamespace(action_type=2,target_idx=0,source_idx=127)],
                           legal_potions=lambda:[])
    assert complete_actions(battle)==(2<<29 | 127,)


def test_multi_bits_preserve_every_selected_hand_slot():
    assert action_bits(SimpleNamespace(action_type=3,target_idx=0,source_idx=0b1000000101))==3<<29 | 0b1000000101

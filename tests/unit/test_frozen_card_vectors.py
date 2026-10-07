import pytest
import torch

from env.core.cards_vocab import CARD_VOCAB
from rl.rules.rule_vector_model import (
    FROZEN_CARD_VECTOR_TABLE_SCHEMA,
    load_frozen_card_vector_table,
)


def test_static_card_vector_table_validates_vocabulary_and_versions(tmp_path):
    vectors = torch.arange(len(CARD_VOCAB) * 2 * 3, dtype=torch.float32).reshape(
        len(CARD_VOCAB), 2, 3)
    checkpoint = tmp_path / 'vectors.pt'
    torch.save(dict(schema=FROZEN_CARD_VECTOR_TABLE_SCHEMA,
                    card_vocab=list(CARD_VOCAB), vectors=vectors), checkpoint)
    table = load_frozen_card_vector_table(
        str(checkpoint), expected_num_cards=len(CARD_VOCAB))
    assert tuple(table.parameters()) == ()
    ids = torch.tensor([11, 25, 321])
    upgrades = torch.tensor([0, 1, 5])
    torch.testing.assert_close(table.versioned_forward(ids, upgrades),
                               vectors[ids, torch.tensor([0, 1, 1])])
    assert 'vectors' in table.state_dict()


def test_static_card_vector_table_rejects_different_card_order(tmp_path):
    checkpoint = tmp_path / 'vectors.pt'
    vocabulary = list(CARD_VOCAB)
    vocabulary[1], vocabulary[2] = vocabulary[2], vocabulary[1]
    torch.save(dict(schema=FROZEN_CARD_VECTOR_TABLE_SCHEMA,
                    card_vocab=vocabulary,
                    vectors=torch.zeros(len(CARD_VOCAB), 2, 4)), checkpoint)
    with pytest.raises(ValueError, match='vocabulary'):
        load_frozen_card_vector_table(
            str(checkpoint), expected_num_cards=len(CARD_VOCAB))

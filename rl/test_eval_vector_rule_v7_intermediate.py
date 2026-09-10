from rl.eval_vector_rule_v7_intermediate import (
    _calculation_blocks,
    _parse_number,
)


def test_parse_lsd_number():
    value, end = _parse_number(
        ["NUM_LSD_BEGIN", "DIGIT_4", "DIGIT_9", "NUM_END"], 0
    )
    assert value == 94
    assert end == 4


def test_parse_calculation_result_register_and_value():
    tokens = [
        "<CALCULATIONS_BEGIN>",
        "CALCULATION_BEGIN", "CALCULATION_SUBTRACT",
        "CALC_FIELD_left", "VALUE_reg_hp",
        "CALC_FIELD_result", "VALUE_reg_after",
        "CALC_VALUE", "NUM_LSD_BEGIN", "DIGIT_4", "DIGIT_9", "NUM_END",
        "CALCULATION_END", "<CALCULATIONS_END>",
    ]
    assert _calculation_blocks(tokens) == [{
        "operation": "SUBTRACT",
        "result_register": "reg_after",
        "values": [94.0],
    }]

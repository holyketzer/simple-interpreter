import pytest

from calc import Interpreter

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param(" 22 + 11 ", 33),
        pytest.param(" 32 - 10 ", 22),
    ]
)
def test_expressions(expression, expected_result):
    res = Interpreter(expression).expr()

    assert expected_result == res

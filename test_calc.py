import pytest

from calc import Interpreter

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param(" 22 + 11 ", 33),
        pytest.param(" 32 - 10 ", 22),
        pytest.param(" 2 * 101 ", 202),
        pytest.param(" 44 / 4 ", 11),
        pytest.param(" 2 + 3 + 4 - 5 + 6 ", 10),
        pytest.param(" 777 ", 777),
        pytest.param("2 + 2 * 2", 8), # wrong!
    ]
)
def test_expressions(expression, expected_result):
    res = Interpreter(expression).expr()

    assert expected_result == res

def test_invalid_expression():
    with pytest.raises(Exception):
        Interpreter("3 +").expr()

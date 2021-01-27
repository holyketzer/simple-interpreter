import pytest

from calc import Lexer, Interpreter

@pytest.mark.parametrize(
    "expression, expected_result",
    [
       pytest.param(" 22 + 11 ", [22, "+", 11]),
       pytest.param("(2 + 2) * 2", ["(", 2, "+", 2, ")", "*", 2]),
    ]
)
def test_lexer(expression, expected_result):
    res = Lexer(expression).get_all_tokens()

    assert expected_result == res

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param(" 22 + 11 ", 33),
        pytest.param(" 32 - 10 ", 22),
        pytest.param(" 2 * 101 ", 202),
        pytest.param(" 44 / 4 ", 11),
        pytest.param(" 2 + 3 + 4 - 5 + 6 ", 10),
        pytest.param(" 777 ", 777),
        pytest.param("2 + 2 * 2", 6),
        pytest.param("2 + 2 * 2 - 33 / 11", 3),
        pytest.param("7 - 3 - 1", 3),
        pytest.param("8 / 4 / 2", 1),
        pytest.param("(2 + 2) * 2", 8),
        pytest.param("2 * (2 + 2)", 8),
        pytest.param("7 + 3 * (10 / (12 / (3 + 1) - 1))", 22),
    ]
)
def test_expressions(expression, expected_result):
    res = Interpreter(Lexer(expression)).evaluate()

    assert expected_result == res

@pytest.mark.parametrize(
    "expression",
    [
       pytest.param("3 + "),
       pytest.param("10 + 2 * 3 2 * 10 / 4 5 8"),
    ]
)
def test_invalid_expression(expression):
    with pytest.raises(Exception, match=r"Invalid syntax"):
        Interpreter(Lexer(expression)).evaluate()

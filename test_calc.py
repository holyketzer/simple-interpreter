import pytest

from calc import Lexer, Parser, Interpreter, ReversePolishNotationTranslator, LISPTranslator

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
        pytest.param("+7 - -3", 10),
    ]
)
def test_expressions(expression, expected_result):
    res = Interpreter(Parser(Lexer(expression))).evaluate()

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
        Interpreter(Parser(Lexer(expression))).evaluate()

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param(" 22 + 11 ", "22 11 +"),
        pytest.param("(5 + 3) * 12 / 3", "5 3 + 12 * 3 /"),
    ]
)
def test_reverse_polish_notation_translator(expression, expected_result):
    res = ReversePolishNotationTranslator(Parser(Lexer(expression))).evaluate()

    assert expected_result == res

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param(" 22 + 11 ", "(+ 22 11)"),
        pytest.param("(5 + 3) * 12 / 3", "(/ (* (+ 5 3) 12) 3)"),
    ]
)
def test_lisp_translator(expression, expected_result):
    res = LISPTranslator(Parser(Lexer(expression))).evaluate()

    assert expected_result == res

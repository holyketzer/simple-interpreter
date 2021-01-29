import pytest

from calc import Lexer, Parser, Interpreter, ReversePolishNotationTranslator, LISPTranslator

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param(" 22 + 11 ", [22, "+", 11]),
        pytest.param("(2 + 2) * 2", ["(", 2, "+", 2, ")", "*", 2]),
        pytest.param("BEGIN a := 2; END.", ["BEGIN", "a", ":=", 2, ";", "END", "."])
    ]
)
def test_lexer(expression, expected_result):
    res = Lexer(expression).get_all_tokens()

    assert expected_result == res

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param("BEGIN a := 2; b := a * 2 END.", "a := 2; b := a * 2"),
        pytest.param(
            '''
            BEGIN
                BEGIN
                    number := 2;
                    a := number;
                    b := 10 * a + 10 * number / 4;
                    c := a - - b
                END;
                x := 11;
            END.
            ''',
            "number := 2; a := number; b := 10 * a + 10 * number / 4; c := a - -b; x := 11; "
        )
    ]
)
def test_parser(expression, expected_result):
    res = Parser(Lexer(expression)).parse()
    print(Lexer(expression).get_all_tokens())

    assert str(res) == expected_result

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
    interpreter = Interpreter(Parser(Lexer(f"BEGIN a := {expression} END.")))
    interpreter.evaluate()

    assert expected_result == interpreter.global_scope['a']

@pytest.mark.parametrize(
    "expression",
    [
       pytest.param("3 + "),
       pytest.param("10 + 2 * 3 2 * 10 / 4 5 8"),
    ]
)
def test_invalid_expression(expression):
    with pytest.raises(Exception, match=r"Invalid syntax"):
        Interpreter(Parser(Lexer(f"BEGIN a := {expression} END."))).evaluate()

# @pytest.mark.parametrize(
#     "expression, expected_result",
#     [
#         pytest.param(" 22 + 11 ", "22 11 +"),
#         pytest.param("(5 + 3) * 12 / 3", "5 3 + 12 * 3 /"),
#     ]
# )
# def test_reverse_polish_notation_translator(expression, expected_result):
#     res = ReversePolishNotationTranslator(Parser(Lexer(f"BEGIN a := {expression} END."))).evaluate()

#     assert expected_result == res

# @pytest.mark.parametrize(
#     "expression, expected_result",
#     [
#         pytest.param(" 22 + 11 ", "(+ 22 11)"),
#         pytest.param("(5 + 3) * 12 / 3", "(/ (* (+ 5 3) 12) 3)"),
#     ]
# )
# def test_lisp_translator(expression, expected_result):
#     res = LISPTranslator(Parser(Lexer(expression))).evaluate()

#     assert expected_result == res

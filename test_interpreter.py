import pytest

from interpreter import Lexer, Parser, Interpreter, SemanticAnalyzer

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param(" 22 + 11 ", [22, "+", 11]),
        pytest.param("(2 + 2) * 2", ["(", 2, "+", 2, ")", "*", 2]),
        pytest.param("BEGIN a := 2; END.", ["begin", "a", ":=", 2, ";", "end", "."]),
        pytest.param('''
            PROGRAM Part10;
            VAR
               number     : INTEGER;
               y          : REAL;
            PROCEDURE P1;
                VAR A : REAL;
            BEGIN {P1}
                a := 1
            END;  {P1}
            BEGIN {Part10}
               BEGIN
                  number := 2;
               END;
               y := 20 / 7 + 3.14;
               { writeln('y = ', y); }
               { writeln('number = ', number); }
            END.  {Part10}
            ''',
            [
                "program", "part10", ";", "var", "number", ":", "integer", ";", "y", ":", "real", ";",
                "procedure", "p1", ";", "var", "a", ":", "real", ";", "begin", "a", ":=", 1, "end", ";",
                "begin", "begin", "number", ":=", 2, ";", "end", ";",
                "y", ":=", 20, "/", 7, "+", 3.14, ";", "end", ".",
            ]
        ),
    ]
)
def test_lexer(expression, expected_result):
    res = Lexer(expression).get_all_tokens()

    assert expected_result == res

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        pytest.param("PROGRAM test; BEGIN a := 2; b := a * 2 END.", "PROGRAM test;  a := 2; b := a * 2."),
        pytest.param(
            '''
            PROGRAM test;
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
            "PROGRAM test;  number := 2; a := number; b := 10 * a + 10 * number / 4; c := a - -b; x := 11; ."
        )
    ]
)
def test_parser(expression, expected_result):
    res = Parser(Lexer(expression)).parse()

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
        pytest.param("9 div 2", 4),
    ]
)
def test_expressions(expression, expected_result):
    tree = Parser(Lexer(f"PROGRAM test; BEGIN a := {expression} END.")).parse()
    interpreter = Interpreter(tree)
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
        tree = Parser(Lexer(f"PROGRAM test; BEGIN a := {expression} END.")).parse()
        Interpreter(tree).evaluate()

def test_parser_case_insensitive():
    sources = '''
        PROGRAM Part10;

        VAR
           number     : INTEGER;
           a, _b, c, x : INTEGER;
           y          : REAL;

        PROCEDURE P1(z, z2: REAL; z3 : integer);
            VAR A2 : REAL;
        BEGIN {P1}
            a := 1
        END;  {P1}

        beGin {Part10}
           BEGIN
              number := 2;
              A := number;
              _b := 10 * a + 10 * number DIV 4;
              C := a - - _b
           END;
           x := 11;
           y := 20 / 7 + 3.14;
           { writeln('a = ', a); }
           { writeln('b = ', _b); }
           { writeln('c = ', c); }
           { writeln('number = ', number); }
           { writeln('x = ', x); }
           { writeln('y = ', y); }
        End.  {Part10}

        '''

    tree = Parser(Lexer(sources)).parse()
    SemanticAnalyzer().visit(tree)

    interpreter = Interpreter(tree)
    interpreter.evaluate()

    assert interpreter.global_scope['a'] == 2
    assert interpreter.global_scope['_b'] == 25
    assert interpreter.global_scope['c'] == 27
    assert interpreter.global_scope['x'] == 11
    assert interpreter.global_scope['y'] == 20 / 7 + 3.14

@pytest.mark.parametrize(
    "sources, expected_error",
    [
        pytest.param(
            '''
            PROGRAM NameError1;
            VAR
               a : INTEGER;

            BEGIN
               a := 2 + b;
            END.
            ''',
            "unknown variable 'b'",
        ),
        pytest.param(
            '''
            PROGRAM NameError2;
            VAR
               b : INTEGER;

            BEGIN
               b := 1;
               a := b + 2;
            END.
            ''',
            "unknown variable 'a'",
        ),
        pytest.param(
            '''
            PROGRAM Part11;
            VAR
                number : INTEGER;
                a, b   : INTEGER;
                y      : REAL;

                procedure p1;
                    VAR a : real;
                begin
                end;

            BEGIN {Part11}
               number := 2;
               a := number ;
               b := 10 * a + 10 * number DIV 4;
               y := 20 / 7 + 3.14
            END.  {Part11}
            ''',
            None,
        ),
        pytest.param(
            '''
            program SymTab6;
               var x, y : integer;
               var y : real;
            begin
               x := x + y;
            end.
            ''',
            "variable 'y' already defined as 'integer'",
        ),
    ]
)
def test_semantic_analyzer(sources, expected_error):
    root = Parser(Lexer(sources)).parse()
    analyzer = SemanticAnalyzer()

    if expected_error:
        with pytest.raises(NameError, match=expected_error):
            analyzer.visit(root)
    else:
        assert analyzer.visit(root) is None

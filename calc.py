# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
PLUS, MINUS, MUL, REAL_DIV, INT_DIV = "PLUS", "MINUS", "MUL", "REAL_DIV", "INT_DIV"
INTEGER, REAL, EOF = "integer", "real", "EOF"
INTEGER_CONST, REAL_CONST = "INTEGER_CONST", "REAL_CONST"
OPEN_PRS, CLOSE_PRS = "(", ")"
LCURLY, RCURLY = "{", "}"
UNDERSCORE = "_"
COLON = ":"
COMMA = ","
PROGRAM, BEGIN, VAR, END, DOT, SEMI, ASSIGN, ID = "program", "begin", "var", "end", "DOT", "SEMI", "ASSIGN", "ID"
COMMENT = "COMMENT"

# Grammar
# expr : product ((PLUS | MINUS) product)*
# product : number ((MUL | DIV | INT_DIV) number)*
# number : (PLUS | MINUS) number | INTEGER | LPAREN expr RPAREN

# Pascal grammar
# ==============
# program : PROGRAM variable SEMI block DOT
# block : declarations compound_statement
# declarations : VAR (variable_declaration SEMI)+ | empty
# variable_declaration : variable (COMMA variable)* COLON type_spec
# type_spec : INTEGER | REAL
# compound_statement : BEGIN statement_list end
# statement_list : statement | statement SEMI statement_list
# statement : compound_statement | assignment_statement | empty
# assignment_statement : variable ASSIGN expr
# expr : product ((PLUS | MINUS) product)*
# product : factor ((MUL | INT_DIV | FLOAT_DIV) factor)*
# factor : PLUS factor | MINUS factor | INTEGER_CONST | REAL_CONST | variable | LPAREN expr RPAREN
# variable : ID
# empty :

class Token:
    def __init__(self, type, value):
        # token type: INTEGER, PLUS, or EOF
        self.type = type
        # token value: 0, 1, 2. 3, 4, 5, 6, 7, 8, 9, '+', or None
        self.value = value

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(INTEGER, 3)
            Token(PLUS '+')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()

class Lexer:
    def __init__(self, text):
        # client string input, e.g. "3+5"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        # current token instance
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception(f"Invalid character '{self.current_char}'")

    def advance(self):
        """Advance the 'pos' pointer and set the 'current_char' variable."""
        self.pos += 1
        self.current_char = self.peek(delta=0)

    def peek(self, delta=1):
        if self.pos + delta < len(self.text):
            return self.text[self.pos + delta]
        else:
            return ''  # Indicates end of input

    def skip_whitespaces(self):
        while self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char != RCURLY:
            self.advance()
        self.advance()  # the closing curly brace

    def number(self):
        """Return a (mulidigit) integer consumed from the input."""
        result = ''

        while self.current_char.isdigit() or self.current_char == ".":
            result += self.current_char
            self.advance()

        if "." in result:
            return Token(REAL_CONST, float(result))
        else:
            return Token(INTEGER_CONST, int(result))

    def is_name_begin(self, current_char):
        return current_char.isalpha() or current_char == UNDERSCORE

    def id(self):
        result = ''
        while self.current_char.isdigit() or self.is_name_begin(self.current_char):
            result += self.current_char
            self.advance()

        result = result.lower().strip()

        if result in (BEGIN, END, PROGRAM, VAR, INTEGER, REAL):
            return Token(result, result)
        elif result == 'div':
            return Token(INT_DIV, 'div')
        else:
            return Token(ID, result)

    def comment(self):
        result = ''
        while self.current_char != RCURLY:
            result += self.current_char
            self.advance()

        result += self.current_char

        return Token(COMMENT, result.strip())

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens.
        """
        self.skip_whitespaces()

        while self.current_char:
            if self.current_char.isdigit():
                return self.number()

            if self.is_name_begin(self.current_char):
                return self.id()

            if self.current_char == LCURLY:
                self.skip_comment()
                self.skip_whitespaces()
                continue

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, "+")

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, "-")

            if self.current_char == '*':
                self.advance()
                return Token(MUL, "*")

            if self.current_char == '/':
                self.advance()
                return Token(REAL_DIV, "/")

            if self.current_char in (OPEN_PRS, CLOSE_PRS):
                current_char = self.current_char
                self.advance()
                return Token(current_char, current_char)

            if self.current_char == ".":
                self.advance()
                return Token(DOT, ".")

            if self.current_char == ",":
                self.advance()
                return Token(COMMA, ",")

            if self.current_char == ";":
                self.advance()
                return Token(SEMI, ";")

            if self.current_char == ":":
                if self.peek() == "=":
                    self.advance()
                    self.advance()
                    return Token(ASSIGN, ":=")
                else:
                    self.advance()
                    return Token(COLON, ":")

            self.error()

        return Token(EOF, None)

    def get_all_tokens(self):
        res = []
        current_token = self.get_next_token()

        while current_token.type != EOF:
            res.append(current_token.value)
            current_token = self.get_next_token()

        return res

class BaseNode(object):
    pass

class NoOpNode(BaseNode):
    def __str__(self):
        return ""

class NumNode(BaseNode):
    def __init__(self, token):
        self.token = token
        self.value = token.value

    def __str__(self):
        return str(self.value)

class VarNode(NumNode):
    pass

class VarDeclNode(BaseNode):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node

    def __str__(self):
        return f"var {self.var_node} : {self.type_node};"

class TypeNode(BaseNode):
    def __init__(self, token):
        self.token = token
        self.name = token.value

    def __str__(self):
        return self.name

class AssignNode(BaseNode):
    def __init__(self, var_node, expr_node):
        self.var_node = var_node
        self.expr_node = expr_node

    def __str__(self):
        return f"{self.var_node} := {self.expr_node}"

class BinOpNode(BaseNode):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

    def __str__(self):
        return f"{self.left} {self.op.value} {self.right}"

class UnaryOpNode(BaseNode):
    def __init__(self, op, child):
        self.token = self.op = op
        self.child = child

    def __str__(self):
        return f"{self.op.value}{self.child}"

class CompoundNode(BaseNode):
    def __init__(self, children):
        self.children = children

    def __str__(self):
        return "; ".join(map(str, self.children))

class BlockNode(BaseNode):
    def __init__(self, declaration_nodes, compound_node):
        self.declaration_nodes = declaration_nodes
        self.compound_node = compound_node

    def __str__(self):
        declarations = " ".join(map(str, self.declaration_nodes))
        return f"{declarations} {self.compound_node}"

class ProgramNode(BaseNode):
    def __init__(self, name, block_node):
        self.name = name
        self.block_node = block_node

    def __str__(self):
        return f"PROGRAM {self.name}; {self.block_node}."

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception(f'Invalid syntax token: {self.current_token}')

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        self.eat(PROGRAM)
        var_node = self.variable()
        self.eat(SEMI)
        block_node = self.block()
        self.eat(DOT)
        return ProgramNode(var_node.value, block_node)

    def block(self):
        declaration_nodes = self.declarations()
        compound_node = self.compound_statement()
        return BlockNode(declaration_nodes, compound_node)

    def declarations(self):
        res = []
        if self.current_token.type == VAR:
            self.eat(VAR)

            while self.current_token.type == ID:
                res += self.variable_declaration()
                self.eat(SEMI)

        return res

    def variable_declaration(self):
        var_nodes = [self.variable()]

        while self.current_token.type == COMMA:
            self.eat(COMMA)
            var_nodes.append(self.variable())

        self.eat(COLON)
        type_node = self.type_spec()

        return list(map(lambda var_node: VarDeclNode(var_node, type_node), var_nodes))

    def type_spec(self):
        token = self.current_token

        if token.type in (INTEGER, REAL):
            self.eat(token.type)
            return TypeNode(token)
        else:
            self.error()

    def compound_statement(self):
        self.eat(BEGIN)
        nodes = self.statement_list()
        self.eat(END)
        return CompoundNode(nodes)

    def statement_list(self):
        nodes_list = [self.statement()]

        while self.current_token.type == SEMI:
            self.eat(SEMI)
            nodes_list += self.statement_list()

        return nodes_list

    def statement(self):
        if self.current_token.type == BEGIN:
            return self.compound_statement()
        elif self.current_token.type == ID:
            return self.assignment_statement()
        else:
            return NoOpNode()

    def assignment_statement(self):
        var_node = self.variable()
        self.eat(ASSIGN)
        expr_node = self.expr()
        return AssignNode(var_node, expr_node)

    def expr(self):
        left = self.product()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            self.eat(token.type)
            left = BinOpNode(left, token, self.product())

        return left

    def product(self):
        left = self.factor()

        while self.current_token.type in (MUL, REAL_DIV, INT_DIV):
            token = self.current_token
            self.eat(token.type)
            left = BinOpNode(left, token, self.factor())

        return left

    def factor(self):
        """Return an INTEGER token value"""
        token = self.current_token

        if token.type == OPEN_PRS:
            self.eat(OPEN_PRS)
            node = self.expr()
            self.eat(CLOSE_PRS)
            return node
        elif token.type in (PLUS, MINUS):
            self.eat(token.type)
            return UnaryOpNode(token, self.factor())
        elif token.type == ID:
            return self.variable()
        elif token.type == REAL_CONST:
            self.eat(REAL_CONST)
            return NumNode(token)
        else:
            self.eat(INTEGER_CONST)
            return NumNode(token)

    def variable(self):
        var_token = self.current_token
        self.eat(ID)
        return VarNode(var_token)

    def parse(self):
        root = self.program()

        if self.current_token.type != EOF:
            self.error()

        return root

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))

class InterpreterWithParser(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser
        self.global_scope = {}

    def evaluate(self):
        root = self.parser.parse()
        return self.visit(root)

class Interpreter(InterpreterWithParser):
    def visit_BinOpNode(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == REAL_DIV:
            return self.visit(node.left) / self.visit(node.right)
        elif node.op.type == INT_DIV:
            return self.visit(node.left) // self.visit(node.right)
        else:
            raise NameError(f"unsupported binary op {node.op}")

    def visit_UnaryOpNode(self, node):
        if node.op.type == MINUS:
            return -self.visit(node.child)
        elif node.op.type == PLUS:
            return +self.visit(node.child)

    def visit_NumNode(self, node):
        return node.value

    def visit_VarNode(self, node):
        if node.value in self.global_scope:
            return self.global_scope[node.value]
        else:
            raise NameError(node.value)

    def visit_NoOpNode(self, node):
        pass

    def visit_AssignNode(self, node):
        self.global_scope[node.var_node.value] = self.visit(node.expr_node)

    def visit_VarDeclNode(self, node):
        self.global_scope[node.var_node.value] = 0

    def visit_CompoundNode(self, node):
        for child in node.children:
            self.visit(child)

    def visit_BlockNode(self, node):
        for declaration_node in node.declaration_nodes:
            self.visit(declaration_node)

        self.visit(node.compound_node)

    def visit_ProgramNode(self, node):
        self.visit(node.block_node)

class ReversePolishNotationTranslator(InterpreterWithParser):
    def visit_BinOpNode(self, node):
        return f"{self.visit(node.left)} {self.visit(node.right)} {node.op.value}"

    def visit_NumNode(self, node):
        return str(node.value)

class LISPTranslator(InterpreterWithParser):
    def visit_BinOpNode(self, node):
        return f"({node.op.value} {self.visit(node.left)} {self.visit(node.right)})"

    def visit_NumNode(self, node):
        return str(node.value)

def main():
    while True:
        try:
            # To run under Python3 replace 'raw_input' call
            # with 'input'
            text = input('calc> ')
        except EOFError:
            break
        if not text:
            continue
        interpreter = Interpreter(Lexer(text))
        result = interpreter.evaluate()
        print(result)


if __name__ == '__main__':
    main()

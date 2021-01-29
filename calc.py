# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
PLUS, MINUS, MUL, DIV = 'PLUS', 'MINUS', 'MUL', 'DIV'
INTEGER, EOF = 'INTEGER', 'EOF'
OPEN_PRS, CLOSE_PRS = "(", ")"
BEGIN, END, DOT, SEMI, ASSIGN, ID = "BEGIN", "END", "DOT", "SEMI", "ASSIGN", "ID"

# Grammar
# expr : product ((PLUS | MINUS) product)*
# product : number ((MUL | DIV) number)*
# number : (PLUS | MINUS) number | INTEGER | LPAREN expr RPAREN

# Pascal grammar
# program : compound_statement DOT
# compound_statement : BEGIN statement_list END
# statement_list : statement | statement SEMI statement_list
# statement : compound_statement | assignment_statement | empty
# assignment_statement : variable ASSIGN expr
# expr : product ((PLUS | MINUS) product)*
# product : factor ((MUL | DIV) factor)*
# factor : PLUS factor | MINUS factor | INTEGER | variable | LPAREN expr RPAREN
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

    def integer(self):
        """Return a (mulidigit) integer consumed from the input."""
        result = ''

        while self.current_char.isdigit():
            result += self.current_char
            self.advance()

        return int(result)

    def id(self):
        result = ''
        while self.current_char.isdigit() or self.current_char.isalpha():
            result += self.current_char
            self.advance()

        if result in (BEGIN, END):
            return Token(result, result)
        else:
            return Token(ID, result)

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens.
        """
        self.skip_whitespaces()

        while self.current_char:
            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

            if self.current_char.isalpha():
                return self.id()

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
                return Token(DIV, "/")

            if self.current_char in (OPEN_PRS, CLOSE_PRS):
                current_char = self.current_char
                self.advance()
                return Token(current_char, current_char)

            if self.current_char == ".":
                self.advance()
                return Token(DOT, ".")

            if self.current_char == ";":
                self.advance()
                return Token(SEMI, ";")

            if self.current_char == ":":
                if self.peek() == "=":
                    self.advance()
                    self.advance()
                    return Token(ASSIGN, ":=")

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

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

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
        node = self.compound_statement()
        self.eat(DOT)
        return node

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

        while self.current_token.type in (MUL, DIV):
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
        else:
            self.eat(INTEGER)
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
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)

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

    def visit_CompoundNode(self, node):
        for child in node.children:
            self.visit(child)

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

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
PLUS, MINUS, MUL, DIV = 'PLUS', 'MINUS', 'MUL', 'DIV'
INTEGER, EOF = 'INTEGER', 'EOF'
OPEN_PRS, CLOSE_PRS = "(", ")"

# Grammar
# expr : product ((PLUS | MINUS) product)*
# product : number ((MUL | DIV) number)*
# number : INTEGER | O_PAREN expr C_PAREN

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
        raise Exception('Invalid character')

    def advance(self):
        """Advance the 'pos' pointer and set the 'current_char' variable."""
        self.pos += 1

        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = ''  # Indicates end of input

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

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens.
        """
        self.skip_whitespaces()

        while self.current_char:
            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

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

class OpNode(BaseNode):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class NumNode(BaseNode):
    def __init__(self, token):
        self.token = token
        self.value = token.value

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

    def number(self):
        """Return an INTEGER token value"""
        token = self.current_token

        if token.type == OPEN_PRS:
            self.eat(OPEN_PRS)
            node = self.expr()
            self.eat(CLOSE_PRS)
            return node
        else:
            self.eat(INTEGER)
            return NumNode(token)

    def product(self):
        left = self.number()

        while self.current_token.type in (MUL, DIV):
            token = self.current_token
            self.eat(token.type)
            left = OpNode(left, token, self.number())

        return left

    def expr(self):
        left = self.product()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            self.eat(token.type)
            left = OpNode(left, token, self.product())

        return left

    def parse(self):
        root = self.expr()

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


class Interpreter(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser

    def visit_OpNode(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_NumNode(self, node):
        return node.value

    def evaluate(self):
        root = self.parser.parse()
        return self.visit(root)

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

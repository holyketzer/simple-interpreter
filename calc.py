# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
PLUS, MINUS, MUL, DIV = 'PLUS', 'MINUS', 'MUL', 'DIV'
INTEGER, EOF = 'INTEGER', 'EOF'

# Grammar
# expr : factor ((PLUS | MINUS) factor)*
# factor : number ((MUL | DIV) number)*
# number : INTEGER

class Token(object):
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


class Interpreter(object):
    def __init__(self, text):
        # client string input, e.g. "3+5"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        # current token instance
        self.current_token = None
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Error parsing input')

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
                return Token(PLUS, self.current_char)

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, self.current_char)

            if self.current_char == '*':
                self.advance()
                return Token(MUL, self.current_char)

            if self.current_char == '/':
                self.advance()
                return Token(DIV, self.current_char)

            self.error()

        return Token(EOF, None)

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error()

    def number(self):
        """Return an INTEGER token value"""
        token = self.current_token
        self.eat(INTEGER)
        return token.value

    def factor(self):
        # self.current_token = self.get_next_token()
        left = self.number()

        while self.current_token.type in (MUL, DIV):
            token = self.current_token
            self.eat(token.type)

            if token.type == MUL:
                left = left * self.number()
            elif token.type == DIV:
                left = left / self.number()

        return left

    def expr(self):
        self.current_token = self.get_next_token()

        left = self.factor()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            self.eat(token.type)

            if token.type == PLUS:
                left = left + self.factor()
            elif token.type == MINUS:
                left = left - self.factor()

        return left

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
        interpreter = Interpreter(text)
        result = interpreter.expr()
        print(result)


if __name__ == '__main__':
    main()

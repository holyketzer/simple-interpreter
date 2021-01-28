WHITESPACES = Set.new([" ", "\r", "\n", "\t"])
NUMBER = "NUM"
PLUS = "+"
MINUS = "-"
MUL = "*"
DIV = "/"
LPAREN = "("
RPAREN = ")"

def is_digit(s)
  48 <= s.ord && s.ord <= 57
end

def is_digit_str(str)
  str.each_char do |c|
    if !is_digit(c)
      return false
    end
  end

  true
end

class Lexer
  def initialize(str)
    @str = str
    @pos = 0
    advance
  end

  def is_whitespace(s)
    WHITESPACES.include?(s)
  end

  def advance
    if @pos < @str.size
      @current_char = @str[@pos]
      @pos += 1
    else
      @current_char = nil
    end
  end

  def skip_whitespaces
    while @current_char && is_whitespace(@current_char) do
      advance
    end
  end

  def next_token
    skip_whitespaces

    if @current_char
      if is_digit(@current_char)
        token = @current_char

        while advance && is_digit(@current_char) do
          token += @current_char
        end

        token
      else
        @current_char.tap { advance }
      end
    end
  end

  def all_tokens
    res = []

    while (token = next_token) do
      res << token
    end

    res
  end
end

# Grammar
# sum :: product ((PLUS | MINUS) product)*
# product :: number ((MUL | DIV) number)*
# number :: integer | LPAREN sum RPAREN

class Parser
  def initialize(str)
    @lexer = Lexer.new(str)
    @current_token = @lexer.next_token
  end

  def eat(token)
    if (token == NUMBER && is_digit_str(@current_token)) || (@current_token == token)
      @current_token = @lexer.next_token
    else
      raise Exception, "expected '#{token}' got '#{@current_token}'"
    end
  end

  def sum
    left = product

    while @current_token == PLUS || @current_token == MINUS
      if @current_token == PLUS
        eat(PLUS)
        left += product
      elsif @current_token == MINUS
        eat(MINUS)
        left -= product
      end
    end

    left
  end

  def product
    left = number

    while @current_token == MUL || @current_token == DIV
      if @current_token == MUL
        eat(MUL)
        left *= number
      elsif @current_token == DIV
        eat(DIV)
        left /= number
      end
    end

    left
  end

  def number
    if @current_token == LPAREN
      eat(LPAREN)
      sum.tap { eat(RPAREN) }
    else
      integer
    end
  end

  def integer
    @current_token.to_i.tap { eat(NUMBER) }
  end

  def parse()
    sum
  end
end

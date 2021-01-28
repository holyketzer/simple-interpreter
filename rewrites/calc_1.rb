class Lexer
  WHITESPACES = Set.new([" ", "\t", "\r", "\n"]).freeze

  attr_reader :current_token

  def initialize(str)
    @str = str
    @current_token = nil
    @offset = 0
    @current_char = @str[@offset]
  end

  def next_token
    skip_whitespaces

    if @current_char && is_digit(@current_char)
      @current_token = @current_char

      while advance && is_digit(@current_char) do
        @current_token += @current_char
      end
    else
      @current_token = @current_char
      advance
    end

    @current_token
  end

  def skip_whitespaces
    while WHITESPACES.include?(@current_char) do
      advance
    end
  end

  def is_digit(s)
    code = s.ord
    48 <= code && code <= 57
  end

  def advance
    if @offset + 1 < @str.size
      @offset += 1
      @current_char = @str[@offset]
    else
      @current_char = nil
    end
  end

  def all_tokens
    res = []

    while next_token do
      res << @current_token
    end

    res
  end
end

class Parser
  def initialize(str)
    @lexer = Lexer.new(str)
  end

  def parse()
    res = @lexer.next_token.to_i

    while @lexer.next_token do
      op = @lexer.current_token

      if op == "+"
        res += @lexer.next_token.to_i
      elsif op == "-"
        res -= @lexer.next_token.to_i
      else
        raise ArgumentError, "unknown operator '#{op}'"
      end
    end

    res
  end
end

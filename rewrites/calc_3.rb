EQ = "="
SEMI = ";"
DOT = "."
PLUS = "+"
MINUS = "-"
MUL = "*"
DIV = "/"
LPAREN = "("
RPAREN = ")"

UNDER = "_"
LCURCLY = "{"
RCURCLY = "}"
COLON = ":"
ASSIGN = ":="
COMMA = ","

INT_DIV = "DIV"
INTEGER = "INTEGER"
REAL = "REAL"
ID = "ID"
INT_CONST = 'INT_CONST'
REAL_CONST = 'REAL_CONST'
PROGRAM = "PROGRAM"
K_BEGIN = "BEGIN"
K_END = "END"
VAR = "VAR"

KEYWORDS = Set.new([K_BEGIN, K_END, VAR, PROGRAM, INTEGER, REAL, INT_DIV])
WHITESPACES = Set.new([" ", "\t", "\r", "\n"])
SYMBOLS = Set.new([SEMI, DOT, PLUS, MINUS, MUL, DIV, LPAREN, RPAREN, COMMA])
TYPES = Set.new([INTEGER, REAL])

# Grammar
# program : PROGRAM variable SEMI block DOT
# block : declarations compound_statement
# declarations : VAR (variable_declaration SEMI)+ | empty
# variable_declaration: variable (COMMA variable)* COLON type_spec
# type_spec : INTEGER | REAL
# compound_statement : BEGIN statement_list END
# statement_list : statement | statement SEMI statement_list
# statement : compound_statement | assignment_statement | empty
# assignment_statement : variable ASSIGN expression
# expression : product ((PLUS | MINUS) product)*
# product : factor ((MUL | INT_DIV | FLOAT_DIV) factor)*
# factor : PLUS expression | MINUS expression | num_const | LPAREN expression RPAREN | variable
# num_const : INTEGER_CONST | REAL_CONST
# variable : ID
# empty :

class String
  def underscore
    self.gsub(/::/, '/').
    gsub(/([A-Z]+)([A-Z][a-z])/,'\1_\2').
    gsub(/([a-z\d])([A-Z])/,'\1_\2').
    tr("-", "_").
    downcase
  end
end

def is_digit(s)
  s && s.ord >= 48 && s.ord <= 57
end

def is_name_start(s)
  s && (s == UNDER || (s >= 'A' && s <= 'Z') || (s >= 'a' && s <= 'z'))
end

def is_name_part(s)
  is_name_start(s) || is_digit(s)
end

class Token
  attr_reader :value, :type

  def initialize(value, type = nil)
    @value = value
    @type = type || value
  end

  def to_s
    "#{type} -> #{value}"
  end
end

class Lexer
  attr_reader :current_char

  def initialize(str)
    @str = str
    @pos = 0
    advance!
  end

  def advance!
    if @pos < @str.size
      @current_char = @str[@pos]
      @pos += 1
    else
      @current_char = nil
    end
  end

  def is_whitespace(s)
    WHITESPACES.include?(s)
  end

  def skip_whitespaces
    while is_whitespace(current_char) do
      advance!
    end
  end

  def skip_comment
    while current_char != RCURCLY do
      advance!
    end
    advance!
  end

  def number_token
    token = ''

    while is_digit(current_char)
      token += current_char
      advance!
    end

    if current_char == DOT
      token += DOT
      advance!

      while is_digit(current_char)
        token += current_char
        advance!
      end

      return Token.new(token.to_f, REAL_CONST)
    else
      return Token.new(token.to_i, INT_CONST)
    end
  end

  def name_token
    token = current_char
    advance!

    while is_name_part(current_char) do
      token += current_char
      advance!
    end

    token = token.upcase

    if KEYWORDS.include?(token)
      Token.new(token)
    else
      Token.new(token, ID)
    end
  end

  def next_token
    skip_whitespaces

    while current_char do
      if current_char == LCURCLY
        skip_comment
        skip_whitespaces
        next
      end

      if is_digit(current_char)
        return number_token
      end

      if is_name_start(current_char)
        return name_token
      end

      if current_char == COLON
        advance!

        if current_char == EQ
          advance!
          return Token.new(ASSIGN)
        end

        return Token.new(COLON)
      end

      if SYMBOLS.include?(current_char)
        return Token.new(current_char).tap { advance! }
      end

      raise Exception.new("unexpected char: '#{current_char}' code=#{current_char&.ord} pos=#{@pos}")
    end
  end

  def all_tokens
    res = []

    while (token = next_token) do
      res << token.value
    end

    return res
  end
end

class BaseNode
end

class NoOpNode < BaseNode
end

class VarNode < BaseNode
  attr_reader :name

  def initialize(name)
    @name = name
  end
end

class NumConstNode < BaseNode
  attr_reader :value

  def initialize(value)
    @value = value
  end
end

class VarDefinitionNode < BaseNode
  attr_reader :var_node
  attr_reader :type_node

  def initialize(var_node, type_node)
    @var_node = var_node
    @type_node = type_node
  end
end

class TypeNode < BaseNode
  attr_reader :type_name

  def initialize(type_name)
    @type_name = type_name
  end
end

class UnaryOpNode < BaseNode
  attr_reader :op
  attr_reader :expr_node

  def initialize(op, expr_node)
    @op = op
    @expr_node = expr_node
  end
end

class BinaryOpNode < BaseNode
  attr_reader :left_node
  attr_reader :op
  attr_reader :right_node

  def initialize(left_node, op, right_node)
    @left_node = left_node
    @op = op
    @right_node = right_node
  end
end

class AssignmentNode < BaseNode
  attr_reader :var_node
  attr_reader :expr_node

  def initialize(var_node, expr_node)
    @var_node = var_node
    @expr_node = expr_node
  end
end

class CompoundStatement < BaseNode
  attr_reader :statement_nodes

  def initialize(statement_nodes)
    @statement_nodes = statement_nodes
  end
end

class BlockNode < BaseNode
  attr_reader :declarations
  attr_reader :compound_node

  def initialize(declarations, compound_node)
    @declarations = declarations
    @compound_node = compound_node
  end
end

class ProgramNode < BaseNode
  attr_reader :name
  attr_reader :block_node

  def initialize(name, block_node)
    @name = name
    @block_node = block_node
  end
end

class Parser
  attr_reader :current_token, :lexer

  def initialize(lexer)
    @lexer = lexer
    @current_token = lexer.next_token
  end

  def parse
    program
  end

  def eat(token_type)
    if current_token.type == token_type
      @current_token = lexer.next_token
    else
      error("expected: '#{token_type}' got: '#{current_token}'")
    end
  end

  def error(msg)
    raise Exception.new(msg)
  end

  def program
    eat(PROGRAM)
    var_node = variable
    eat(SEMI)
    block_node = block
    eat(DOT)

    ProgramNode.new(var_node.name, block_node)
  end

  def block
    BlockNode.new(declarations, compound_statement)
  end

  def declarations
    if current_token.type == VAR
      eat(VAR)

      vars = []

      while current_token.type == ID do
        vars += variable_declaration
        eat(SEMI)
      end

      CompoundStatement.new(vars)
    else
      NoOpNode.new
    end
  end

  def variable_declaration
    var_nodes = [variable]

    while current_token.type == COMMA do
      eat(COMMA)
      var_nodes << variable
    end

    eat(COLON)

    type_node = type_spec

    var_nodes.map { |var_node| VarDefinitionNode.new(var_node, type_node) }
  end

  def type_spec
    if TYPES.include?(current_token.type)
      TypeNode.new(current_token.type).tap { eat(current_token.type) }
    end
  end

  def compound_statement
    eat(K_BEGIN)
    CompoundStatement.new(statement_list).tap { eat(K_END) }
  end

  def statement_list
    statements = [statement]

    while current_token.type == SEMI
      eat(SEMI)
      statements << statement
    end

    statements
  end

  def statement
    if current_token.type == K_BEGIN
      compound_statement
    elsif current_token.type == ID
      assignment_statement
    else
      NoOpNode.new
    end
  end

  def assignment_statement
    var_node = variable
    eat(ASSIGN)
    expr_node = expression
    AssignmentNode.new(var_node, expr_node)
  end

  def expression
    left = product

    while current_token.type == PLUS || current_token.type == MINUS do
      op = current_token.type
      eat(current_token.type)
      left = BinaryOpNode.new(left, op, product)
    end

    left
  end

  def product
    left = factor

    while current_token.type == MUL || current_token.type == INT_DIV || current_token.type == DIV do
      op = current_token.type
      eat(current_token.type)
      left = BinaryOpNode.new(left, op, factor)
    end

    left
  end

  def factor
    if current_token.type == PLUS
      eat(PLUS)
      UnaryOpNode.new(PLUS, expression)
    elsif current_token.type == MINUS
      eat(MINUS)
      UnaryOpNode.new(MINUS, expression)
    elsif current_token.type == LPAREN
      eat(LPAREN)
      expression.tap { eat(RPAREN) }
    elsif current_token.type == ID
      variable
    elsif current_token.type == INT_CONST || current_token.type == REAL_CONST
      num_const
    end
  end

  def num_const
    NumConstNode.new(current_token.value).tap { eat(current_token.type) }
  end

  def variable
    VarNode.new(current_token.value).tap { eat(ID) }
  end
end

class BaseInterpreter
  def evaluate_node(node)
    send("evaluate_#{node.class.name.underscore}", node)
  end
end

class Interpreter < BaseInterpreter
  def initialize(parser)
    @parser = parser
    @global_scope = {}
  end

  def evaluate()
    root = @parser.parse
    evaluate_node(root)
    @global_scope
  end

  def evaluate_no_op_node(node)
  end

  def evaluate_var_node(node)
    if @global_scope.include?(node.name)
      @global_scope[node.name]
    else
      raise Exception.new("unknown variable: '#{node.name}'")
    end
  end

  def evaluate_num_const_node(node)
    node.value
  end

  def evaluate_var_definition_node(node)
    @global_scope[node.var_node.name] = nil
  end

  def evaluate_type_node(node)
  end

  def evaluate_unary_op_node(node)
    if node.op == PLUS
      evaluate_node(node.expr_node)
    elsif node.op == MINUS
      -evaluate_node(node.expr_node)
    else
      raise Exception.new("unknown op: '#{node.op}'")
    end
  end

  def evaluate_binary_op_node(node)
    if node.op == PLUS
      evaluate_node(node.left_node) + evaluate_node(node.right_node)
    elsif node.op == MINUS
      evaluate_node(node.left_node) - evaluate_node(node.right_node)
    elsif node.op == MUL
      evaluate_node(node.left_node) * evaluate_node(node.right_node)
    elsif node.op == DIV
      evaluate_node(node.left_node) / evaluate_node(node.right_node)
    elsif node.op == INT_DIV
      evaluate_node(node.left_node).to_i / evaluate_node(node.right_node).to_i
    else
      raise Exception.new("unknown op: '#{node.op}'")
    end
  end

  def evaluate_assignment_node(node)
    @global_scope[node.var_node.name] = evaluate_node(node.expr_node)
  end

  def evaluate_compound_statement(node)
    node.statement_nodes.each { |child| evaluate_node(child) }
  end

  def evaluate_block_node(node)
    evaluate_node(node.declarations)
    evaluate_node(node.compound_node)
  end

  def evaluate_program_node(node)
    evaluate_node(node.block_node)
  end
end

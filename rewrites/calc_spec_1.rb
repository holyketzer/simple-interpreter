require './calc_1.rb'

RSpec.describe Lexer do
  subject { lexer.all_tokens  }
  let(:lexer) { described_class.new(expression) }
  let(:expression) { "1 + 23 / 4" }

  it { is_expected.to eq ["1", "+", "23", "/", "4"] }
end

RSpec.describe Parser do
  subject { parser.parse() }

  let(:parser) { described_class.new(expression) }
  let(:expression) { "7 - 3 + 2 - 1" }

  it { is_expected.to eq 5 }
end

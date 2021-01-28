require './calc.rb'

RSpec.describe Lexer do
  subject { lexer.all_tokens  }
  let(:lexer) { described_class.new(expression) }

  context 'with spaces' do
    let(:expression) { "1 + 23 / 4" }

    it { is_expected.to eq ["1", "+", "23", "/", "4"] }
  end

  context 'without spaces' do
    let(:expression) { "1+23/4" }

    it { is_expected.to eq ["1", "+", "23", "/", "4"] }
  end

  context 'without spaces parentheses' do
    let(:expression) { "1+23/4-(2*(3+4))" }

    it { is_expected.to eq ["1", "+", "23", "/", "4", "-", "(", "2", "*", "(", "3", "+", "4", ")", ")"] }
  end
end

RSpec.describe Parser do
  subject { parser.parse() }

  let(:parser) { described_class.new(expression) }

  context "simple expression" do
    let(:expression) { "7 - 3 + 2 - 1" }

    it { is_expected.to eq 5 }
  end

  context "priority expression" do
    let(:expression) { "2 + 2 * 2" }

    it { is_expected.to eq 6 }
  end

  context "expression with parentheses" do
    let(:expression) { "(2 + 2) * 2" }

    it { is_expected.to eq 8 }
  end

  context "complex expression with parentheses" do
    let(:expression) { "(1 + (2 + 2) * 2) * 3" }

    it { is_expected.to eq 27 }
  end
end

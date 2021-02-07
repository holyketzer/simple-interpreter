require './calc_3.rb'

RSpec.describe Lexer do
  subject { lexer.all_tokens  }
  let(:lexer) { described_class.new(expression) }

  context 'with spaces' do
    let(:expression) { "1 + 23 / 4" }

    it { is_expected.to eq [1, "+", 23, "/", 4] }
  end

  context 'without spaces' do
    let(:expression) { "1+23/4" }

    it { is_expected.to eq [1, "+", 23, "/", 4] }
  end

  context 'without spaces parentheses' do
    let(:expression) { "1+23/4-(2*(3+4.45))" }

    it { is_expected.to eq [1, "+", 23, "/", 4, "-", "(", 2, "*", "(", 3, "+", 4.45, ")", ")"] }
  end

  context 'with Pascal source' do
    let(:expression) do
      <<~PAS
        PROGRAM Part10;
        VAR
           number     : INTEGER;
           y          : REAL;
        BEGIN {Part10}
           BEGIN
              number := 2;
           END;
           y := 20 / 7 + 3.14;
           { writeln('y = ', y); }
           { writeln('number = ', number); }
        END.  {Part10}
      PAS
    end

    it do
      is_expected.to eq [
        "PROGRAM", "PART10", ";", "VAR", "NUMBER", ":", "INTEGER", ";", "Y", ":", "REAL", ";",
        "BEGIN", "BEGIN", "NUMBER", ":=", 2, ";", "END", ";",
        "Y", ":=", 20, "/", 7, "+", 3.14, ";", "END", ".",
      ]
    end
  end
end

RSpec.describe Interpreter do
  subject do
    interpreter = Interpreter.new(Parser.new(Lexer.new(sources)))
    interpreter.evaluate()
  end

  let(:sources) do
    <<~PAS
      PROGRAM Part10;
      VAR
         number     : INTEGER;
         a, _b, c, x : INTEGER;
         y          : REAL;

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
    PAS
  end

  it do
    expect(subject['A']).to eq 2
    expect(subject['_B']).to eq 25
    expect(subject['C']).to eq 27
    expect(subject['X']).to eq 11
    expect(subject['Y']).to eq 20 / 7 + 3.14
  end

  describe 'simple expressions' do
    let(:sources) { "PROGRAM test; BEGIN a := #{expression} END." }

    context "simple expression" do
      let(:expression) { "7 - 3 + 2 - 1" }

      it { expect(subject['A']).to eq 5 }
    end

    context "priority expression" do
      let(:expression) { "2 + 2 * 2" }

      it { expect(subject['A']).to eq 6 }
    end

    context "expression with parentheses" do
      let(:expression) { "(2 + 2) * 2" }

      it { expect(subject['A']).to eq 8 }
    end

    context "complex expression with parentheses" do
      let(:expression) { "(1 + (2 + 2) * 2) * 3" }

      it { expect(subject['A']).to eq 27 }
    end
  end
end

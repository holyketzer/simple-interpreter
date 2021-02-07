PROGRAM Part11;
VAR
   number : INTEGER;
   a, b   : INTEGER;
   y      : REAL;

PROCEDURE p1(z : INTEGER);
   var x : INTEGER;
begin
   y := a + b + z;
end;

VAR
   nnn : INTEGER;

PROCEDURE p2(b : REAL);
   var x2 : INTEGER;

   PROCEDURE p22(x : INTEGER);
      var xx : INTEGER;
   begin
      x2 := x;
      p1(x2 + 1);
   end;
begin
   a := 1;
   p22(x2 * 3);
end;

BEGIN {Part11}
   {number := 2;}
   number := 20 / 7 + 3.14;
   p2(5);
   a := number;
   {WriteLn(a);}
   b := 10 * a + 10 * number DIV 4;
END.  {Part11}

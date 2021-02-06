PROGRAM Part11;
VAR
   number : INTEGER;
   a, b   : INTEGER;
   y      : REAL;

PROCEDURE p1;
   var x : INTEGER;
begin
   x := 1;
end;

VAR
   nnn : INTEGER;

PROCEDURE p2(b : REAL);
   var x2 : INTEGER;

   PROCEDURE p22(x : INTEGER);
      var xx : INTEGER;
   begin
      xx := x;
   end;
begin
   x2 := 1;
end;

BEGIN {Part11}
   {number := 2;}
   a := number;
   {WriteLn(a);}
   b := 10 * a + 10 * number DIV 4;
   y := 20 / 7 + 3.14
END.  {Part11}

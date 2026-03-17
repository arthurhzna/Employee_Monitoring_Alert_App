class Cecko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0

    def a(self) -> int:
        return self._a & 8589934591  

    def b(self) -> int:
        return self._c & 511

    def c(self) -> int:
        return self._b & 63

    def d(self, var1: int) -> None:
        self._a = var1 & 8589934591 

    def e(self, var1: int) -> None:
        self._c = var1 & 511

    def f(self, var1: int) -> None:
        self._b = var1 & 63


class Acko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: int = 0
        self._f: int = 0
        self._g: int = 0
        self._h: int = 0
        self._i: int = 0
        self._j: Cecko = None

    def a(self) -> int:
        return self._i & 1

    def b(self) -> int:
        return self._a & 255

    def c(self) -> int:
        return self._b & 1

    def d(self) -> int:
        return self._d & 1

    def e(self) -> int:
        return self._f & 1

    def f(self) -> Cecko:
        return self._j

    def g(self) -> int:
        return self._e & 1

    def h(self) -> int:
        return self._c & 1

    def i(self) -> int:
        return self._g & 1

    def j(self) -> int:
        return self._h & 1

    def k(self, var1: int) -> None:
        self._i = var1 & 1

    def l(self, var1: int) -> None:
        self._a = var1 & 255

    def m(self, var1: int) -> None:
        self._b = var1 & 1

    def n(self, var1: int) -> None:
        self._d = var1 & 1

    def o(self, var1: int) -> None:
        self._f = var1 & 1

    def p(self, var1: Cecko) -> None:
        self._j = var1

    def q(self, var1: int) -> None:
        self._e = var1 & 1

    def r(self, var1: int) -> None:
        self._c = var1 & 1

    def s(self, var1: int) -> None:
        self._g = var1 & 1

    def t(self, var1: int) -> None:
        self._h = var1 & 1


class Becko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: int = 0
        self._f: int = 0
        self._g: int = 0
        self._h: int = 0

    def a(self) -> int:
        return self._g & 3

    def b(self) -> int:
        return self._h & 15

    def c(self) -> int:
        return self._e & 8191

    def d(self) -> int:
        return self._c & 1

    def e(self) -> int:
        return self._a & 255

    def f(self) -> int:
        return self._b & 1

    def g(self) -> int:
        return self._d & 1

    def h(self) -> int:
        return self._f & 3

    def i(self, var1: int) -> None:
        self._g = var1 & 3

    def j(self, var1: int) -> None:
        self._h = var1 & 15

    def k(self, var1: int) -> None:
        self._e = var1 & 8191

    def l(self, var1: int) -> None:
        self._c = var1 & 1

    def m(self, var1: int) -> None:
        self._a = var1 & 255

    def n(self, var1: int) -> None:
        self._b = var1 & 1

    def o(self, var1: int) -> None:
        self._d = var1 & 1

    def p(self, var1: int) -> None:
        self._f = var1 & 3


class Decko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: int = 0
        self._f: int = 0
        self._g: int = 0
        self._h: int = 0
        self._i: int = 0
        self._j: Cecko = None
        self._k: int = 0
        self._l: list = []  # datovej typ třída 'e'
        self._m: int = 0

    def a(self) -> int:
        return self._i & 1

    def b(self) -> int:
        return self._k & 255

    def c(self) -> int:
        return self._d & 3

    def d(self) -> int:
        return self._g & 3

    def e(self) -> int:
        return self._e & 4095

    def f(self) -> int:
        return self._j & 255

    def g(self) -> int:
        return self._b & 1

    def h(self) -> int:
        return self._a & 255

    def i(self) -> int:
        # return self._f & '\uffff'
        return self._f & 65535

    def j(self) -> int:
        return self._h & 31

    def k(self) -> int:
        return self._c & 1

    def l(self, var1: int) -> None:
        self._m = var1 & -1

    def m(self, var1: int) -> None:
        self._i = var1 & 1

    def n(self, var1: int) -> None:
        self._k = var1 & 255

    def o(self, var1: list) -> None:
        self._l = var1

    def p(self, var1: int) -> None:
        self._d = var1 & 3

    def q(self, var1: int) -> None:
        self._g = var1 & 3

    def r(self, var1: int) -> None:
        self._e = var1 & 4095

    def s(self, var1: int) -> None:
        self._j = var1 & 255

    def t(self, var1: int) -> None:
        self._b = var1 & 1

    def u(self, var1: int) -> None:
        self._a = var1 & 255

    def v(self, var1: int) -> None:
        # self._f = var1 & '\uffff'
        self._f = var1 & 65535

    def w(self, var1: int) -> None:
        self._h = var1 & 31

    def x(self, var1: int) -> None:
        self._c = var1 & 1


class Ecko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0

    def a(self) -> int:
        return self._c & 8191

    def b(self) -> int:
        # return self._a & '\uffff'
        return self._a & 65535

    def c(self) -> int:
        return self._b & 7

    def d(self, var1: int) -> None:
        self._c = var1 & 8191

    def e(self, var1: int) -> None:
        # self._a = var1 & '\uffff'
        self._a = var1 & 65535

    def f(self, var1: int) -> None:
        self._b = var1 & 7


class Efko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: int = 0
        self._f: int = 0
        self._g: int = 0
        self._h: int = 0
        self._i: int = 0
        self._j: int = 0
        self._k: int = 0
        self._l: int = 0
        self._m: int = 0
        self._n: int = 0
        self._o: int = 0
        self._p: int = 0
        self._q: int = 0

    def A(self, var1: int) -> None:
        self._o = var1 & 1

    def B(self, var1: int) -> None:
        self._p = var1 & 1

    def C(self, var1: int) -> None:
        self._q = var1 & 255

    def D(self, var1: int) -> None:
        # self._c = var1 & '\uffff'
        self._c = var1 & 65535

    def E(self, var1: int) -> None:
        self._f = var1 & 1

    def F(self, var1: int) -> None:
        self._e = var1 & 3

    def G(self, var1: int) -> None:
        self._j = var1 & 3

    def H(self, var1: int) -> None:
        self._b = var1 & 255

    def a(self) -> int:
        return self._n & 1

    def b(self) -> int:
        return self._h & 1

    def c(self) -> int:
        return self._m & 1

    def d(self) -> int:
        return self._g & 1

    def e(self) -> int:
        return self._l & 1

    def f(self) -> int:
        return self._k & 1

    def g(self) -> int:
        return self._d & 3

    def h(self) -> int:
        return self._i & 1

    def i(self) -> int:
        return self._a & 16777215

    def j(self) -> int:
        return self._o & 1

    def k(self) -> int:
        return self._p & 1

    def l(self) -> int:
        return self._q & 255

    def m(self) -> int:
        # return self._c & '\uffff'
        return self._c & 65535

    def n(self) -> int:
        return self._f & 1

    def o(self) -> int:
        return self._e & 3

    def p(self) -> int:
        return self._j & 3

    def q(self) -> int:
        return self._b & 255

    def r(self, var1: int) -> None:
        self._n = var1 & 1

    def s(self, var1: int) -> None:
        self._h = var1 & 1

    def t(self, var1: int) -> None:
        self._m = var1 & 1

    def u(self, var1: int) -> None:
        self._g = var1 & 1

    def v(self, var1: int) -> None:
        self._l = var1 & 1

    def w(self, var1: int) -> None:
        self._k = var1 & 1

    def x(self, var1: int) -> None:
        self._d = var1 & 3

    def y(self, var1: int) -> None:
        self._i = var1 & 1

    def z(self, var1: int) -> None:
        self._a = var1 & 16777215


class Gecko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: int = 0
        self._f: int = 0
        self._g: int = 0
        self._h: int = 0
        self._i: int = 0
        self._j: int = 0
        self._k: int = 0
        self._l: int = 0
        self._m: int = 0
        self._n: int = 0
        self._o: int = 0
        self._p: list = []  # list třídy 'h'
        self._q: int = 0

    def A(self, var1: int) -> None:
        self._e = var1 & 4095

    def B(self, var1: int) -> None:
        self._j = var1 & 255

    def C(self, var1: int) -> None:
        self._b = var1 & 1

    def D(self, var1: int) -> None:
        self._a = var1 & 255

    def E(self, var1: int) -> None:
        self._h = var1 & 31

    def F(self, var1: int) -> None:
        self._c = var1 & 1

    def a(self) -> int:
        return self._i & 1

    def b(self) -> int:
        return self._k & 255

    def c(self) -> int:
        return self._m & 8191

    def d(self) -> int:
        return self._o & 4095

    def e(self) -> int:
        # return self._f & '\uffff'
        return self._f & 65535

    def f(self) -> int:
        return self._d & 3

    def g(self) -> int:
        return self._g & 3

    def h(self) -> int:
        return self._l & 7

    def i(self) -> int:
        return self._n & 15

    def j(self) -> int:
        return self._e & 4095

    def k(self) -> int:
        return self._j & 255

    def l(self) -> int:
        return self._b & 1

    def m(self) -> int:
        return self._a & 255

    def n(self) -> int:
        return self._h & 31

    def o(self) -> int:
        return self._c & 1

    def p(self, var1: int) -> None:
        self._q = var1 & -1

    def q(self, var1: int) -> None:
        self._i = var1 & 1

    def r(self, var1: int) -> None:
        self._k = var1 & 255

    def s(self, var1: int) -> None:
        self._m = var1 & 8191

    def t(self, var1: list) -> None:
        self._p = var1

    def u(self, var1: int) -> None:
        self._o = var1 & 4095

    def v(self, var1: int) -> None:
        # self._f = var1 & '\uffff'
        self._f = var1 & 65535

    def w(self, var1: int) -> None:
        self._d = var1 & 3

    def x(self, var1: int) -> None:
        self._g = var1 & 3

    def y(self, var1: int) -> None:
        self._l = var1 & 7

    def z(self, var1: int) -> None:
        self._n = var1 & 15


class Hacko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: int = 0

    def a(self) -> int:
        return self._e & 4095

    def b(self) -> int:
        return self._c & 8191

    def c(self) -> int:
        return self._b & 7

    def d(self) -> int:
        return self._d & 15

    def e(self) -> int:
        return self._a & 255

    def f(self, var1: int) -> None:
        self._e = var1 & 4095

    def g(self, var1: int) -> None:
        self._c = var1 & 8191

    def h(self, var1: int) -> None:
        self._b = var1 & 7

    def i(self, var1: int) -> None:
        self._d = var1 & 15

    def j(self, var1: int) -> None:
        self._a = var1 & 255


class Icko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: int = 0
        self._f: int = 0
        self._g: int = 0

    def getB(self) -> int:
        return self._b

    def getD(self) -> int:
        return self._d

    def getF(self) -> int:
        return self._f

    def a(self) -> int:
        return self._a & 15

    def b(self) -> int:
        return self._b & 7

    def c(self) -> int:
        return self._d & 32767

    def d(self) -> int:
        return self._f & 32767

    def e(self) -> int:
        return self._c & 1

    def f(self) -> int:
        return self._e & 1

    def g(self) -> int:
        return self._g & 1

    def h(self, var1: int) -> None:
        self._a = var1 & 15

    def i(self, var1: int) -> None:
        self._b = var1 & 7

    def j(self, var1: int) -> None:
        self._d = var1 & 32767

    def k(self, var1: int) -> None:
        self._f = var1 & 32767

    def l(self, var1: int) -> None:
        self._c = var1 & 1

    def m(self, var1: int) -> None:
        self._e = var1 & 1

    def n(self, var1: int) -> None:
        self._g = var1 & 1
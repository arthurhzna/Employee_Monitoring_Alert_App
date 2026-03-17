from externals.speaker_tapo_cctv_handlers.util import *
import enum
import externals.speaker_tapo_cctv_handlers.g as g


class Becko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: int = 0

    def a(self) -> int:
        return self._b

    def b(self) -> int:
        return self._a

    def c(self) -> int:
        return self._d

    def d(self) -> int:
        return self._c

    def e(self, var1: int) -> None:
        self._e = var1

    def f(self, var1: int) -> None:
        self._b = var1

    def g(self, var1: int) -> None:
        self._a = var1

    def h(self, var1: int) -> None:
        self._d = var1

    def i(self, var1: int) -> None:
        self._c = var1


class Cecko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: int = 0
        self._e: list = []

    def a(self) -> int:
        return self._b

    def b(self) -> int:
        return self._a

    def c(self) -> list:
        return self._e

    def d(self) -> int:
        return self._c

    def e(self) -> int:
        return self._d

    def f(self, var1: int) -> None:
        self._b = var1

    def g(self, var1: int) -> None:
        self._a = var1

    def h(self, var1: int) -> None:
        self._c = var1

    def i(self, var1: int) -> None:
        self._d = var1


class Acko:
    def __init__(self):
        self._a: int = 0
        self._b: int = 0
        self._c: int = 0
        self._d: Cecko = None

    def a(self) -> int:
        return self._b

    def b(self) -> int:
        return self._a

    def c(self) -> Cecko:
        return self._d

    def d(self, var1: int) -> None:
        self._b = var1

    def e(self, var1: int) -> None:
        self._a = var1

    def f(self, var1: Cecko) -> None:
        self._d = var1

    def g(self, var1: int) -> None:
        self._c = var1


class Decko:
    a = [0 for i in range(256)]

class Codec(enum.Enum):
    MUXTS_CODEC_HEVC = 1
    MUXTS_CODEC_AVC = 2
    MUXTS_CODEC_AAC = 3
    MUXTS_CODEC_PCMA = 4


class Paketovadlo:
    def __init__(self):
        var1 = Acko()
        var1.e(0)
        var1.d(0)
        var1.g(0)
        var1.f(Cecko())

        self._a = [0 for i in range(188)]
        self._b = 0
        self._c = [0 for i in range(512000)] 
        self._d = var1
        self._g = 0
        self._h = []
        self._i = []
        self._j = False

    def c(self, var1: int, var2: int, var3: int) -> int:
        return byte_overflow(var1 << var2 | var3)

    def d(self, var1: int, var2: int, var3: int) -> int:
        return int_overflow(var1 << var2 | var3)

    def e(self, var1: int, var2: int, var3: int) -> int:
        return long_overflow(var1 << var2 | var3)

    def f(self, var1: int) -> list:
        return [
            byte_overflow(var1 >> 24 & 255),
            byte_overflow(var1 >> 16 & 255),
            byte_overflow(var1 >> 8 & 255),
            byte_overflow(var1 & 255),
        ]

    def g(self, var1: int) -> list:
        return [
            byte_overflow(int_overflow(var1 >> 56 & 255)),
            byte_overflow(int_overflow(var1 >> 48 & 255)),
            byte_overflow(int_overflow(var1 >> 40 & 255)),
            byte_overflow(int_overflow(var1 >> 32 & 255)),
            byte_overflow(int_overflow(var1 >> 24 & 255)),
            byte_overflow(int_overflow(var1 >> 16 & 255)),
            byte_overflow(int_overflow(var1 >> 8 & 255)),
            byte_overflow(int_overflow(var1 & 255)),
        ]

    def h(self, var1: int) -> int:
        return var1 + 1 & 15

    def i(self, var1: list, var2: int, var3: int) -> int:
        # var4 = 255
        var4 = -1
        for var5 in range(var2, var2 + var3):
            var6 = var1[var5]
            var4 = Decko.a[(var4 >> 24 ^ var6) & 255] ^ var4 << 8

        return var4

    def j(self, var1: int) -> int:
        if self._b + var1 > 188:
            return -4
        else:
            for i in range(var1):
                self._a[self._b + i] = -1  
            self._b += var1
            return 0

    def o(self, var1: int) -> None:
        _i = self.i(self._a, var1, self._b - var1)
        _f = self.f(_i)
        ArrayCopy(_f, 0, self._a, self._b, 4)
        self._b += 4

    def p(self) -> None:
        if self._b == 188:
            self._c += self._a[:188]  
            self._b = 0

    def q(self, var1: int, var2: bool, var3: object) -> None:
        var4 = g.Acko()
        var4.l(var1)
        var4.m(0)
        var4.n(0)

        if var2:
            var4.q(255)
            var4.r(0)
            var4.p(var3)
        else:
            var4.q(0)
            var4.r(0)

        if 1 == var1:
            var4.r(0)

        var4.o(0)
        var4.s(0)
        var4.t(0)
        var4.k(0)

        var5 = self.c(0, 0, var4.b())
        var8 = self._a
        var1 = self._b
        var8[var1] = var5
        self._b = var1 + 1

        if var4.b() != 0:
            var5 = self.c(
                self.c(
                    self.c(
                        self.c(
                            self.c(
                                self.c(
                                    self.c(self.c(0, 0, var4.c()), 1, var4.h()),
                                    1,
                                    var4.d(),
                                ),
                                1,
                                var4.g(),
                            ),
                            1,
                            var4.e(),
                        ),
                        1,
                        var4.i(),
                    ),
                    1,
                    var4.j(),
                ),
                1,
                var4.a(),
            )
            var8 = self._a
            var1 = self._b
            var8[var1] = var5
            self._b = var1 + 1

            if var2:
                var6 = 0
                var5 = byte_overflow(
                    int_overflow(var4.f().a() >> 25 & 255 | var6)
                )
                var8 = self._a
                var1 = self._b
                var8[var1] = var5
                self._b = var1 + 1
                ArrayCopy(
                    self.f(
                        self.d((var6 + (var4.f().a() & 33554431)), 6, var4.f().c()) << 1
                        | var4.f().b() & 256
                    ),
                    0,
                    self._a,
                    self._b,
                    4,
                )
                self._b += 4
                var5 = var4.f().b() | 0
                var8 = self._a
                var1 = self._b
                var8[var1] = var5
                self._b = var1 + 1
            else:
                self.j(var4.b() - 1)

    def s(self, var1: int, var2: int, var3: int, var4: int) -> None:
        var5 = g.Becko()
        var5.m(71)
        var5.n(0)
        var5.l(var2)
        var5.o(0)
        var5.k(var1)
        var5.p(0)
        var5.i(var3)
        var5.j(var4)
        ArrayCopy(
            self.f(
                self.d(
                    self.d(
                        self.d(
                            self.d(
                                self.d(
                                    self.d(
                                        self.d(self.d(0, 0, var5.e()), 1, var5.f()),
                                        1,
                                        var5.d(),
                                    ),
                                    1,
                                    var5.g(),
                                ),
                                13,
                                var5.c(),
                            ),
                            2,
                            var5.h(),
                        ),
                        2,
                        var5.a(),
                    ),
                    4,
                    var5.b(),
                )
            ),
            0,
            self._a,
            self._b,
            4,
        )
        self._b = 4
        if 1 == var2 and 1 == var3:
            self._a[4] = 0
            self._b = 4 + 1

    def t(self, var1: list, var2: bool) -> None:
        if var1 is not None and len(var1) > 0:
            var3 = g.Decko()
            var3.u(0)
            var3.t(255)
            var3.x(0)
            var3.p(255)

            var4 = 0
            var5 = 0
            if var2:
                var4 = 9
                var5 = 5
            else:
                var4 = 8
                var5 = 4

            var3.r(var4 + len(var1) * 4)
            var3.v(1)
            var3.q(255)
            var3.w(0)
            var3.m(255)
            var3.s(0)
            var3.n(0)
            var3.o(var1)
            var3.l(1)
            ArrayCopy(
                self.g(
                    self.e(
                        self.e(
                            self.e(
                                self.e(
                                    self.e(
                                        self.e(
                                            self.e(
                                                self.e(
                                                    self.e(
                                                        self.e(
                                                            self.e(0, 0, var3.h()),
                                                            1,
                                                            var3.g(),
                                                        ),
                                                        1,
                                                        var3.k(),
                                                    ),
                                                    2,
                                                    var3.c(),
                                                ),
                                                12,
                                                var3.e(),
                                            ),
                                            16,
                                            var3.i(),
                                        ),
                                        2,
                                        var3.d(),
                                    ),
                                    5,
                                    var3.j(),
                                ),
                                1,
                                var3.a(),
                            ),
                            8,
                            var3.f(),
                        ),
                        8,
                        var3.b(),
                    )
                ),
                0,
                self._a,
                self._b,
                8,
            )
            self._b += 8

            for cosi in var1:
                ArrayCopy(
                    self.f(
                        self.d(
                            self.d(self.d(0, 0, cosi.b()), 3, cosi.c()), 13, cosi.a()
                        )
                    ),
                    0,
                    self._a,
                    self._b,
                    4,
                )
                self._b += 4

            self.o(var5)
            self.j(188 - self._b)
            self.p()

    # var5 == g.Icko
    def u(self, var1: int, var2: int, var3: int, var4: int, var5: object) -> None:
        var6 = g.Efko()
        var6.z(1)
        var6.H(var2)
        var6.D(var1)
        var6.x(2)
        var6.F(0)
        var6.E(0)
        var6.u(0)
        var6.s(0)
        var6.y(0)
        var6.G(var3)
        var6.w(0)
        var6.v(0)
        var6.t(0)
        var6.r(0)
        var6.A(0)
        var6.B(0)
        var6.C(var4)
        ArrayCopy(
            self.g(
                self.e(
                    self.e(
                        self.e(
                            self.e(
                                self.e(
                                    self.e(
                                        self.e(
                                            self.e(
                                                self.e(
                                                    self.e(
                                                        self.e(
                                                            self.e(
                                                                self.e(
                                                                    self.e(
                                                                        self.e(
                                                                            self.e(
                                                                                0,
                                                                                0,
                                                                                var6.i(),
                                                                            ),
                                                                            8,
                                                                            var6.q(),
                                                                        ),
                                                                        16,
                                                                        var6.m(),
                                                                    ),
                                                                    2,
                                                                    var6.g(),
                                                                ),
                                                                2,
                                                                var6.o(),
                                                            ),
                                                            1,
                                                            var6.n(),
                                                        ),
                                                        1,
                                                        var6.d(),
                                                    ),
                                                    1,
                                                    var6.b(),
                                                ),
                                                1,
                                                var6.h(),
                                            ),
                                            2,
                                            var6.p(),
                                        ),
                                        1,
                                        var6.f(),
                                    ),
                                    1,
                                    var6.e(),
                                ),
                                1,
                                var6.c(),
                            ),
                            1,
                            var6.a(),
                        ),
                        1,
                        var6.j(),
                    ),
                    1,
                    var6.k(),
                )
            ),
            0,
            self._a,
            self._b,
            8,
        )
        self._b += 8
        var7 = byte_overflow(var6.l() | 0)
        var8 = self._a
        var1 = self._b
        var8[var1] = var7
        self._b = var1 + 1
        if 2 == var6.p() and 5 == var6.l():
            var7 = self.c(self.c(self.c(0, 0, var5.a()), 3, var5.b()), 1, var5.e())
            var8 = self._a
            var1 = self._b
            var8[var1] = var7
            self._b = var1 + 1
            ArrayCopy(
                self.f(
                    self.d(
                        self.d(
                            self.d(self.d(0, 0, var5.c()), 1, var5.f()), 15, var5.d()
                        ),
                        1,
                        var5.g(),
                    )
                ),
                0,
                self._a,
                self._b,
                4,
            )
            self._b += 4

        if 3 == var6.p() and 10 == var6.l():
            var5.h(3)
            var7 = self.c(self.c(self.c(0, 0, var5.a()), 3, var5.b()), 1, var5.e())
            var10 = self._a
            var1 = self._b
            var10[var1] = var7
            self._b = var1 + 1
            ArrayCopy(
                self.f(
                    self.d(
                        self.d(
                            self.d(self.d(0, 0, var5.c()), 1, var5.f()), 15, var5.d()
                        ),
                        1,
                        var5.g(),
                    )
                ),
                0,
                self._a,
                self._b,
                4,
            )
            self._b += 4

        if 224 == var2 and 27 == self._d.c().c()[self._f].d():
            var9 = self._a
            var9[self._b] = 0
            self._b += 1
            var9[self._b] = 0
            self._b += 1
            var9[self._b] = 0
            self._b += 1
            var9[self._b] = 1
            self._b += 1
            var9[self._b] = 9
            self._b += 1
            var9[self._b] = -16
            self._b += 1

    def v(self, var1: list, var2: int, var3: bool) -> None:
        if var1 is not None and len(var1) > 0:
            var4 = g.Gecko()
            var4.D(2)
            var4.C(1)
            var4.F(0)
            var4.w(255)
            var5 = 0
            var6 = 0
            if var3:
                var5 = 13
                var6 = 5
            else:
                var5 = 12
                var6 = 4

            var4.A(var5 + len(var1) * 5)
            var4.v(var2)
            var4.x(255)
            var4.E(0)
            var4.q(255)
            var4.B(0)
            var4.r(0)
            var4.y(255)
            var4.s(256)
            var4.z(255)
            var4.u(0)
            var4.t(var1)
            var4.p(1)
            ArrayCopy(
                self.g(
                    self.e(
                        self.e(
                            self.e(
                                self.e(
                                    self.e(
                                        self.e(
                                            self.e(
                                                self.e(
                                                    self.e(
                                                        self.e(
                                                            self.e(0, 0, var4.m()),
                                                            1,
                                                            var4.l(),
                                                        ),
                                                        1,
                                                        var4.o(),
                                                    ),
                                                    2,
                                                    var4.f(),
                                                ),
                                                12,
                                                var4.j(),
                                            ),
                                            16,
                                            var4.e(),
                                        ),
                                        2,
                                        var4.g(),
                                    ),
                                    5,
                                    var4.n(),
                                ),
                                1,
                                var4.a(),
                            ),
                            8,
                            var4.k(),
                        ),
                        8,
                        var4.b(),
                    )
                ),
                0,
                self._a,
                self._b,
                8,
            )
            self._b += 8
            ArrayCopy(
                self.f(
                    self.d(
                        self.d(
                            self.d(self.d(0, 0, var4.h()), 13, var4.c()), 4, var4.i()
                        ),
                        12,
                        var4.d(),
                    )
                ),
                0,
                self._a,
                self._b,
                4,
            )
            self._b += 4

            for cosi in var1:
                var8 = self.c(0, 0, cosi.e())
                var10 = self._a
                var10[self._b] = var8
                self._b += 1
                ArrayCopy(
                    self.f(
                        self.d(
                            self.d(
                                self.d(self.d(0, 0, cosi.c()), 13, cosi.b()),
                                4,
                                cosi.d(),
                            ),
                            12,
                            cosi.a(),
                        )
                    ),
                    0,
                    self._a,
                    self._b,
                    4,
                )
                self._b += 4

            self.o(var6)
            self.j(188 - self._b)
            self.p()

    def a(self, var1: int, var2: int) -> None:
        self._d.c().g(var1)
        self._d.c().f(0)
        self._d.c().h(0)
        self._d.c().i(var2)

    def b(self, var1: int, codec: Codec) -> None:
        var3 = Becko()
        var3.g(var1)
        var3.f(0)
        var1 = codec.value
        if var1 != 1:
            if var1 != 2:
                if var1 != 3:
                    if var1 == 4:
                        var3.h(192)
                        var3.e(17)
                        var3.i(144)
                        self._e = len(self._d.c().c())
                else:
                    var3.h(192)
                    var3.e(17)
                    var3.i(15)
                    self._e = len(self._d.c().c())

            else:
                var3.h(224)
                var3.e(18)
                var3.i(27)
                self._f = len(self._d.c().c())

        else:
            var3.h(224)
            var3.e(18)
            var3.i(36)
            self._f = len(self._d.c().c())

        var1 = self._d.c().d()
        self._d.c().h(var1 + 1)
        self._d.c().c().append(var3)

    def k(self, var1: list, var2: int, var3: int) -> list:
        if var1 is None:
            return None
        else:
            var5 = var1
            if len(var1) > var2:
                var5 = ArraysCopyOf(var1, var2)

            self.n()
            self.a(66, 1)
            var6 = Codec.MUXTS_CODEC_PCMA
            self.b(68, var6)
            self.r(var5, var3, var6)
            return self.l()

    def l(self) -> list:
        var1 = len(self._c)
        if var1 > 0:
            var2 = [x for x in self._c]
            return var2
        else:
            return None

    def m(self) -> None:
        self._c = None

    def n(self) -> None:
        self._b = 0
        self._c = []
        self._d.e(0)
        self._d.d(0)
        self._d.g(0)
        self._d.f(Cecko())
        self._g = 0
        self._h = []
        self._i = []

    def r(self, var1: list, var2: int, var4: Codec) -> None:
        if var1 is not None:
            var5 = 0
            var6 = 0
            var7 = 0
            var8 = 0
            var9 = 0
            var14 = False

            outer_break = False
            while True:  
                while True:  
                    var5 = len(var1)
                    if Codec.MUXTS_CODEC_HEVC != var4 and Codec.MUXTS_CODEC_AVC != var4:
                        if Codec.MUXTS_CODEC_AAC == var4:
                            var6 = self._e
                            var7 = var5 + 8
                            break

                        if Codec.MUXTS_CODEC_PCMA != var4:
                            var14 = False
                            var7 = 0
                            var8 = 0
                            var9 = 0
                            outer_break = True
                            break

                        var7 = self._e
                    else:
                        var7 = self._f

                    var6 = var7
                    var7 = var5
                    break

                if outer_break:
                    break

                var9 = var6
                var8 = var7
                var7 = 0
                var14 = False
                break

            while True:
                while var5 > 0:
                    var13 = Becko()
                    var15 = 0
                    if self._g == 0:
                        var12 = g.Ecko()
                        var12.e(self._d.c().e())
                        var12.d(self._d.c().b())
                        var12.f(255)
                        self._h.append(var12)

                        for var15 in range(self._d.c().d()):
                            var16 = g.Hacko()
                            var13 = self._d.c().c()[var15]
                            var16.g(var13.b())
                            var16.j(var13.d())
                            var16.f(0)
                            var16.h(255)
                            var16.i(255)
                            self._i.append(var16)

                    if self._g % 64 == 0:
                        self.s(self._d.b(), 1, 1, self._d.a())
                        self.t(self._h, True)
                        var15 = self.h(self._d.a())
                        self._d.d(var15)
                        self._g += 1
                        self.s(self._d.c().b(), 1, 1, self._d.c().a())
                        self.v(self._i, self._d.c().e(), True)
                        var15 = self.h(self._d.c().a())
                        self._d.c().f(var15)
                        self._g += 1
                    else:
                        var17 = None
                        if var14:
                            var17 = self._d.c().c()[var9]
                            var20 = []
                            if var5 >= 184:
                                self.s(var17.b(), 0, 1, var17.a())
                                var20 = self._a
                                var15 = self._b
                                ArrayCopy(var1, var7, var20, var15, 188 - var15)
                                var15 = self._b
                                var5 -= 188 - var15
                                self._b = var15 + (188 - var15)
                                var7 += 188 - var15
                            else:
                                self.s(var17.b(), 0, 3, var17.a())
                                self.q(188 - var5 - 5, False, g.Cecko())
                                var20 = self._a
                                var15 = self._b
                                ArrayCopy(var1, var7, var20, var15, 188 - var15)
                                var15 = self._b
                                var7 += 188 - var15
                                var5 -= 188 - var15
                                self._b = var15 + (188 - var15)

                            var17.f(self.h(var17.a()))
                            self._g += 1
                            self.p()
                        else:
                            var17 = self._d.c().c()[var9]
                            self.s(var17.b(), 1, 3, var17.a())
                            var18 = g.Cecko()
                            var18.d(var2)
                            var18.f(255)
                            var18.e(0)
                            if self._j:
                                self.q(7, True, var18)
                            elif (
                                Codec.MUXTS_CODEC_AAC != var4
                                and Codec.MUXTS_CODEC_PCMA != var4
                            ):
                                self.q(7, True, var18)
                            else:
                                self.q(1, False, var18)

                            var19 = g.Icko()
                            var19.h(2)
                            var19.l(255)
                            var19.m(255)
                            var19.n(255)
                            var19.i(int_overflow(var2 >> 30))
                            var19.j(int_overflow(var2 >> 15))
                            var19.k(var2)
                            var13 = self._d.c().c()[var9]
                            self.u(var8, var13.c(), 2, 5, var19)
                            var13.f(self.h(var13.a()))
                            self._g += 1
                            var6 = self._b
                            if var5 >= 188 - var6:
                                ArrayCopy(var1, var7, self._a, var6, 188 - var6)
                                var6 = self._b
                                var7 += 188 - var6
                                var5 -= 188 - var6
                                self._b = var6 + (188 - var6)

                            self.p()
                            var14 = True
                return
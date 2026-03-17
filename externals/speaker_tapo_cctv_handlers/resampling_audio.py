from externals.speaker_tapo_cctv_handlers.util import short_overflow

class Resampling_Audio:
    _A = [255, 511, 1023, 2047, 4095, 8191, 16383, 32767]

    @staticmethod
    def a(var0: list, var1: int) -> list:
        var2 = [None] * (var1 << 1)

        for var3 in range(var1):
            var4 = Resampling_Audio.c(var0[var3])
            var5 = var3 * 2
            var2[var5] = var4
            var2[var5 + 1] = var4 >> 8

        return var2

    @staticmethod
    def b(var0: list, var1: int) -> list:
        var2 = var1 // 2
        var3 = [None] * var2

        for var1 in range(var2):
            var4 = var1 * 2
            var5 = var0[var4]

            _x = (var0[var4 + 1] & 255) << 8
            _y = var5 & 255
            _z = _x | _y
            _z = short_overflow(_z)
            var3[var1] = Resampling_Audio.d(_z)

        return var3

    @staticmethod
    def c(var0: int) -> int:
        var1 = var0 ^ 85
        var2 = (var1 & 15) << 4
        var4 = (var1 & 112) >> 4
        var5 = 0
        if var4 != 0:
            if var4 != 1:
                var5 = (var2 + 264) << var4 - 1
            else:
                var5 = var2 + 264
        else:
            var5 = var2 + 8

        var3 = var5
        if (var1 & 128) == 0:
            var3 = -var3

        return var3

    @staticmethod
    def d(var0: int) -> int:
        var1 = 0
        if var0 >= 0:
            var1 = 213
        else:
            var2 = 85
            var3 = -var0 - 1
            var1 = var2
            var0 = var3
            if var3 < 0:
                var0 = 32767
                var1 = var2

        var5 = Resampling_Audio.e(var0, Resampling_Audio._A, 8)
        var6 = 0
        if var5 >= 8:
            var6 = var1 ^ 127
        else:
            var4 = var5 << 4  
            var7 = 0
            if var5 < 2:
                var7 = var0 >> 4
            else:
                var7 = var0 >> var5 + 3

            var6 = (var7 & 15 | var4) ^ var1  

        return var6

    @staticmethod
    def e(var0: int, var1: list, var2: int) -> int:
        for var3 in range(var2):
            if var0 <= var1[var3]:
                return var3
        return var2

    @staticmethod
    def f(var0: list) -> list:
        var1 = len(var0)
        var2 = [None] * (var1 << 1)

        for var3 in range(var1):
            var4 = var3 * 2
            var2[var4] = var0[var3]
            var2[var4 + 1] = var0[var3] >> 8

        return var2
def ArrayCopy(
    src_arr: list, src_pos: int, dest_arr: list, dest_pos: int, length: int
) -> None:
    for i in range(length):
        dest_arr[dest_pos + i] = src_arr[src_pos + i]


def ArraysCopyOf(original: list, new_len: int) -> list:
    if new_len <= len(original):
        return original[:new_len]
    else:
        return original[:] + ([0] * (new_len - len(original)))

def byte_overflow(byte):
    if byte > 127:
        return byte - (2 * 128)
    elif byte < -128:
        return byte + (2 * 128)
    else:
        return byte


def short_overflow(short):
    if short > 32767:
        return short - (2 * 32768)
    elif short < -32768:
        return short + (2 * 32768)
    else:
        return short

def int_overflow(int_):
    if int_ > 2147483647:
        return int_ - (2 * 2147483648)
    elif int_ < -2147483648:
        return int_ + (2 * 2147483648)
    else:
        return int_


def long_overflow(long_):
    if long_ > 9223372036854775807:
        return long_ - (2 * 9223372036854775808)
    elif long_ < -9223372036854775808:
        return long_ + (2 * 9223372036854775808)
    else:
        return long_
import operator
import math

__version__ = "2.1.0"


m = [
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]
]

m_inv = [
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505]
]

# Hard-coded D65 illuminant
refX = 0.95047
refY = 1.00000
refZ = 1.08883
refU = 0.19784
refV = 0.46834
lab_e = 0.008856
lab_k = 903.3


# Public API



def husl_to_hex(h, s, l):
    return rgb_to_hex(husl_to_rgb(h, s, l))




def hex_to_husl(hex):
    return rgb_to_husl(*hex_to_rgb(hex))


def huslp_to_rgb(h, s, l):
    return lch_to_rgb(*huslp_to_lch([h, s, l]))


def huslp_to_hex(h, s, l):
    return rgb_to_hex(huslp_to_rgb(h, s, l))


def rgb_to_huslp(r, g, b):
    return lch_to_huslp(rgb_to_lch(r, g, b))


def hex_to_huslp(hex):
    return rgb_to_huslp(*hex_to_rgb(hex))








def _hrad_extremum(L):
    lhs = (math.pow(L, 3.0) + 48.0 * math.pow(L, 2.0) + 768.0 * L + 4096.0) / 1560896.0
    rhs = 1107.0 / 125000.0
    sub = lhs if lhs > rhs else 10.0 * L / 9033.0
    chroma = float("inf")
    result = None
    for row in m:
        for limit in (0.0, 1.0):
            [m1, m2, m3] = row
            top = -3015466475.0 * m3 * sub + 603093295.0 * m2 * sub - 603093295.0 * limit
            bottom = 1356959916.0 * m1 * sub - 452319972.0 * m3 * sub
            hrad = math.atan2(top, bottom)
            # This is a math hack to deal with tan quadrants, I'm too lazy to figure
            # out how to do this properly
            if limit == 0.0:
                hrad += math.pi
            test = max_chroma(L, math.degrees(hrad))
            if test < chroma:
                chroma = test
                result = hrad
    return result


def max_chroma_pastel(L):
    H = math.degrees(_hrad_extremum(L))
    return max_chroma(L, H)












def rgb_prepare(triple):
    ret = []
    for ch in triple:
        ch = round(ch, 3)

        if ch < -0.0001 or ch > 1.0001:
            raise Exception(f"Illegal RGB value {ch:f}")

        if ch < 0:
            ch = 0
        if ch > 1:
            ch = 1

        # Fix for Python 3 which by default rounds 4.5 down to 4.0
        # instead of Python 2 which is rounded to 5.0 which caused
        # a couple off by one errors in the tests. Tests now all pass
        # in Python 2 and Python 3
        ret.append(int(round(ch * 255 + 0.001, 0)))

    return ret


def hex_to_rgb(hex):
    if hex.startswith('#'):
        hex = hex[1:]
    r = int(hex[0:2], 16) / 255.0
    g = int(hex[2:4], 16) / 255.0
    b = int(hex[4:6], 16) / 255.0
    return [r, g, b]


def rgb_to_hex(triple):
    [r, g, b] = triple
    return '#%02x%02x%02x' % tuple(rgb_prepare([r, g, b]))


















def huslp_to_lch(triple):
    H, S, L = triple

    if L > 99.9999999:
        return [100, 0.0, H]
    if L < 0.00000001:
        return [0.0, 0.0, H]

    mx = max_chroma_pastel(L)
    C = mx / 100.0 * S

    return [L, C, H]


def lch_to_huslp(triple):
    L, C, H = triple

    if L > 99.9999999:
        return [H, 0.0, 100.0]
    if L < 0.00000001:
        return [H, 0.0, 0.0]

    mx = max_chroma_pastel(L)
    S = C / mx * 100.0

    return [H, S, L]
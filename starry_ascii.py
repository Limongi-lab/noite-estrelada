import sys
import numpy as np
import pygame

COLS, ROWS = 140, 70
CW, CH = 7, 11
RAMP = " .`'^:;,-~_+<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYXZO0Qo*#MW&8%B@$"
N = len(RAMP)
W, H = COLS * CW, ROWS * CH
AS = W / H

X, Y = np.meshgrid(np.linspace(0, 1, COLS), np.linspace(0, 1, ROWS))
X = X.astype(np.float32)
Y = Y.astype(np.float32)

STOPS = np.array([0.0, 0.3, 0.55, 0.78, 0.92, 1.0])
PAL = np.array([
    [12, 25, 85],
    [30, 65, 150],
    [60, 120, 190],
    [130, 180, 210],
    [190, 215, 190],
    [235, 230, 150],
], dtype=np.float32)

STARS = [
    (63, 20, 28), (143, 15, 18), (146, 87, 22), (216, 18, 25),
    (260, 32, 15), (203, 163, 18), (22, 229, 14), (75, 241, 20),
    (217, 268, 38), (388, 43, 30), (449, 115, 28),
]
STARS = [(x / 630, y / 505, r / 505) for x, y, r in STARS]

WINDOWS = [
    (433, 436), (302, 441), (465, 466), (517, 450), (375, 462),
    (333, 430), (345, 443), (262, 432), (405, 480), (450, 467),
]
WX = np.array([int(x / 630 * (COLS - 1)) for x, y in WINDOWS])
WY = np.array([int(y / 505 * (ROWS - 1)) for x, y in WINDOWS])
WPH = np.random.default_rng(3).uniform(0, 6.28, len(WINDOWS))

rng = np.random.default_rng(7)


def gradient(p):
    out = np.empty(p.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(p, STOPS, PAL[:, c])
    return out


def mix(img, g, color):
    g = np.clip(g, 0, 1)[..., None]
    return img * (1 - g) + np.array(color, dtype=np.float32) * g


def ridge(xs, ys):
    return np.interp(X, np.array(xs) / 630, np.array(ys) / 505)


def rect(x0, y0, x1, y1):
    return (X >= x0 / 630) & (X <= x1 / 630) & (Y >= y0 / 505) & (Y <= y1 / 505)


def build_static():
    img = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    mask = np.zeros((ROWS, COLS), dtype=bool)
    tex = 0.5 + 0.5 * np.sin(X * 95 + Y * 40) * np.sin(Y * 80 - X * 30)

    def put(m, color, amp=0.4):
        nonlocal mask
        c = np.array(color, dtype=np.float32)
        img[m] = c * ((1 - amp / 2) + amp * tex[m][:, None])
        mask = mask | m

    far = ridge([0, 70, 150, 230, 330, 430, 520, 630], [368, 378, 372, 358, 338, 330, 322, 300])
    put(Y > far, (52, 92, 168))

    mt = ridge([480, 505, 540, 575, 600, 630], [335, 310, 290, 265, 256, 262])
    put((X > 0.76) & (Y > mt), (45, 55, 100), 0.6)

    rh = ridge([430, 500, 560, 630], [395, 368, 345, 330])
    put((X > 0.66) & (Y > rh), (50, 105, 150), 0.5)

    lower = ridge([0, 100, 200, 300, 400, 500, 630], [420, 418, 415, 402, 396, 400, 395])
    put(Y > lower, (28, 50, 90), 0.5)

    put((X > 0.68) & (Y > 0.74) & (Y < 0.92), (40, 95, 110), 0.9)

    houses = [
        (252, 420, 292, 445, (120, 90, 80), (150, 70, 55)),
        (300, 432, 332, 450, (60, 70, 100), (80, 90, 120)),
        (420, 428, 448, 448, (70, 85, 120), (110, 120, 150)),
        (356, 452, 396, 475, (90, 95, 120), (115, 120, 145)),
        (384, 470, 432, 502, (110, 115, 140), (140, 145, 165)),
        (440, 458, 482, 482, (70, 90, 125), (100, 115, 145)),
        (516, 446, 548, 462, (60, 85, 115), (90, 110, 140)),
    ]
    for x0, y0, x1, y1, wall, roof in houses:
        put(rect(x0, y0, x1, y1), wall, 0.2)
        put(rect(x0, y0, x1, y0 + (y1 - y0) * 0.4), roof, 0.2)

    put(rect(318, 410, 352, 442), (150, 165, 190), 0.2)
    put(rect(312, 402, 356, 412), (85, 75, 100), 0.2)
    put(rect(350, 388, 364, 410), (165, 180, 205), 0.2)
    hw = (Y - 318 / 505) / (70 / 505) * 8 / 630
    put((Y >= 318 / 505) & (Y <= 388 / 505) & (np.abs(X - 357 / 630) <= hw), (170, 185, 215), 0.1)

    put((X < 0.2) & (Y > 0.88), (80, 110, 45), 0.8)
    return img, mask


STATIC_IMG, STATIC_MASK = build_static()


def sky(t):
    wx = X + 0.012 * np.sin(Y * 20 + t * 0.8)
    wy = Y + 0.012 * np.sin(X * 16 - t * 0.6)

    def spiral(cx, cy, k, sp, fo):
        dx = (wx - cx) * AS
        dy = wy - cy
        r = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)
        return np.sin(2 * th + r * k - t * sp) * np.exp(-r * fo)

    a1 = spiral(0.49, 0.366, 34, 1.2, 3.2)
    a2 = spiral(0.706, 0.465, 70, -1.5, 9)
    strokes = np.sin(wy * 45 + wx * 18 + 3 * np.sin(wx * 9 + t * 0.3))
    bc = np.interp(X, [0, 0.1, 0.3, 0.5, 0.65, 0.8, 1.0], [0.60, 0.68, 0.64, 0.60, 0.54, 0.48, 0.42])
    band = np.exp(-((Y - bc - 0.03 * np.sin(X * 14 - t)) / 0.055) ** 2)
    md = np.hypot((X - 0.905) * AS, Y - 0.139)
    mglow = np.exp(-(md / 0.2) ** 2)

    p = 0.30 + 0.22 * a1 + 0.22 * a2 + 0.5 * band + 0.08 * strokes + 0.35 * mglow + 0.1 * (Y - 0.3)
    img = gradient(np.clip(p, 0, 1))

    for i, (sx, sy, ss) in enumerate(STARS):
        d = np.hypot((X - sx) * AS, Y - sy)
        pulse = 0.88 + 0.12 * np.sin(t * 2 + i * 1.7)
        halo = np.exp(-(d / (ss * 1.6)) ** 2) * 0.35
        ring = 0.8 * np.exp(-((d - ss * 0.85) / (ss * 0.22)) ** 2) * (0.7 + 0.3 * np.sin(d * 90 - t * 3))
        core = np.exp(-(d / (ss * 0.45)) ** 2)
        img = mix(img, halo * pulse, (200, 225, 235))
        img = mix(img, ring * pulse, (235, 245, 240))
        img = mix(img, core * pulse, (255, 230, 100))

    d1 = np.hypot((X - 0.895) * AS, Y - 0.135)
    d2 = np.hypot((X - 0.925) * AS, Y - 0.165)
    outer = np.clip((0.075 - d1) / 0.012, 0, 1)
    cut = np.clip((0.058 - d2) / 0.012, 0, 1)
    ring = 0.3 * np.exp(-((d1 - 0.105) / 0.012) ** 2) * (0.6 + 0.4 * np.sin(t * 2))
    img = mix(img, ring, (240, 230, 120))
    img = mix(img, outer * cut, (230, 150, 50))
    img = mix(img, outer * (1 - cut), (250, 235, 100))
    return img


def cypress(img, t):
    ys = np.array([35, 100, 200, 300, 400, 470, 505]) / 505
    xl = np.array([118, 105, 92, 85, 75, 65, 60]) / 630
    xr = np.array([125, 130, 142, 160, 195, 250, 290]) / 630
    L = np.interp(Y, ys, xl) + 0.004 * np.sin(Y * 30 + t * 2)
    R = np.interp(Y, ys, xr) + 0.004 * np.sin(Y * 26 - t * 2.2)
    tree = (X >= L) & (X <= R) & (Y >= 0.07)
    xc = np.interp(Y, [298 / 505, 420 / 505], [178 / 630, 205 / 630])
    hw = np.interp(Y, [298 / 505, 350 / 505, 420 / 505], [0, 12 / 630, 28 / 630])
    lobe = (Y >= 298 / 505) & (Y <= 0.85) & (np.abs(X - xc) <= hw)
    tree = tree | lobe
    s = 0.5 + 0.5 * np.sin(Y * 55 + (X - L) * 160 + t * 0.6)
    col = np.stack([14 + 70 * s ** 2, 46 + 40 * s, 32 + 8 * s], axis=-1)
    return np.where(tree[..., None], col, img)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Starry Night - ASCII")
    font = pygame.font.SysFont("consolas,couriernew,dejavusansmono,monospace", 11, bold=True)
    clock = pygame.time.Clock()
    cache = {}
    pos = [(c * CW, r * CH) for r in range(ROWS) for c in range(COLS)]
    jitter = rng.integers(-1, 2, size=(ROWS, COLS))
    t = 0.0
    frame = 0
    paused = False

    while True:
        dt = clock.tick(30) / 1000
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()
                if e.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            t += dt
            frame += 1
            if frame % 3 == 0:
                jitter = rng.integers(-1, 2, size=(ROWS, COLS))

        img = sky(t)
        img = np.where(STATIC_MASK[..., None], STATIC_IMG, img)
        bright = 0.6 + 0.4 * np.sin(t * 3 + WPH)
        img[WY, WX] = np.array([255, 215, 100], dtype=np.float32) * bright[:, None]
        img = cypress(img, t)

        lum = (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114) / 255
        idx = np.clip((lum ** 0.8) * (N - 1) + jitter, 0, N - 1).astype(int)
        col = np.clip(img * 1.3 + 20, 0, 255)
        q = np.clip((col / 51).astype(int), 0, 5)
        qi = q[..., 0] * 36 + q[..., 1] * 6 + q[..., 2]
        keys = (idx * 216 + qi).ravel().tolist()

        blits = []
        for k, p in zip(keys, pos):
            if k < 216:
                continue
            s = cache.get(k)
            if s is None:
                i, c = divmod(k, 216)
                color = ((c // 36) * 45 + 30, ((c // 6) % 6) * 45 + 30, (c % 6) * 45 + 30)
                s = font.render(RAMP[i], True, color)
                cache[k] = s
            blits.append((s, p))

        screen.fill((0, 0, 0))
        screen.blits(blits)
        pygame.display.flip()


if __name__ == "__main__":
    main()
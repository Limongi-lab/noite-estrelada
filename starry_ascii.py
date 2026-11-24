import sys
import numpy as np
import pygame

COLS, ROWS = 120, 62
CW, CH = 8, 13
RAMP = " .`'^:;,-~_+<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYXZO0Qo*#MW&8%B@$"
N = len(RAMP)
W, H = COLS * CW, ROWS * CH
AS = W / H

X, Y = np.meshgrid(np.linspace(0, 1, COLS), np.linspace(0, 1, ROWS))
X = X.astype(np.float32)
Y = Y.astype(np.float32)

STOPS = np.array([0.0, 0.35, 0.6, 0.85, 1.0])
PAL = np.array([
    [10, 20, 70],
    [30, 60, 140],
    [70, 120, 190],
    [150, 190, 210],
    [235, 215, 120],
], dtype=np.float32)

STARS = [
    (0.22, 0.10, 0.05), (0.36, 0.07, 0.04), (0.47, 0.20, 0.045),
    (0.60, 0.12, 0.05), (0.70, 0.32, 0.04), (0.30, 0.34, 0.045),
    (0.78, 0.45, 0.035), (0.12, 0.50, 0.035),
]

rng = np.random.default_rng(7)
LX = rng.uniform(0.28, 0.98, 60)
LY = rng.uniform(0.80, 0.97, 60)
LIX = (LX * (COLS - 1)).astype(int)
LIY = (LY * (ROWS - 1)).astype(int)
LPH = rng.uniform(0, 6.28, 60)


def gradient(p):
    out = np.empty(p.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(p, STOPS, PAL[:, c])
    return out


def swirl(wx, wy, cx, cy, k, speed, falloff, t):
    dx = (wx - cx) * AS
    dy = wy - cy
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    return np.sin(th * 2 + r * k - t * speed) * np.exp(-r * falloff)


def paint(t):
    wx = X + 0.03 * np.sin(Y * 14 + t * 0.8)
    wy = Y + 0.03 * np.sin(X * 12 - t * 0.6)

    a1 = swirl(wx, wy, 0.52, 0.30, 18, 1.4, 2.5, t)
    a2 = swirl(wx, wy, 0.28, 0.26, 16, -1.1, 3.0, t)
    band = np.sin(wy * 22 + wx * 6 - t * 0.7) * 0.3
    p = 0.42 + 0.35 * (a1 + a2) + 0.15 * band + Y * 0.35
    img = gradient(np.clip(p, 0, 1))

    for i, (sx, sy, ss) in enumerate(STARS):
        d = np.hypot((X - sx) * AS, Y - sy)
        pulse = 0.85 + 0.15 * np.sin(t * 2 + i)
        core = np.exp(-(d / (ss * 0.35)) ** 2)
        halo = np.exp(-(d / ss) ** 2) * 0.6
        ring = 0.25 * np.exp(-((d - ss * 1.1) / (ss * 0.25)) ** 2) * (0.5 + 0.5 * np.sin(d * 60 - t * 3))
        g = np.clip((core + halo + ring) * pulse, 0, 1)[..., None]
        img = img * (1 - g) + np.array([255, 235, 140], dtype=np.float32) * g

    d1 = np.hypot((X - 0.86) * AS, Y - 0.16)
    d2 = np.hypot((X - 0.885) * AS, Y - 0.14)
    glow = np.exp(-(d1 / 0.14) ** 2) * 0.5
    rings = 0.2 * np.exp(-((d1 - 0.10) / 0.015) ** 2) * (0.6 + 0.4 * np.sin(t * 2))
    moon = np.clip((0.07 - d1) / 0.01, 0, 1) * np.clip((d2 - 0.06) / 0.01, 0, 1)
    g = np.clip(glow + rings + moon, 0, 1)[..., None]
    img = img * (1 - g) + np.array([255, 215, 90], dtype=np.float32) * g

    ridge = 0.66 + 0.035 * np.sin(X * 7 + 1) + 0.015 * np.sin(X * 19)
    m = (Y > ridge)[..., None]
    img = np.where(m, np.array([35, 55, 115], dtype=np.float32), img)

    steeple = (np.abs(X - 0.612) < 0.007) & (Y > 0.58) & (Y < 0.72)
    tip = (Y > 0.52) & (Y <= 0.58) & (np.abs(X - 0.612) < (Y - 0.52) * 0.12)
    img = np.where((steeple | tip)[..., None], np.array([20, 30, 70], dtype=np.float32), img)

    ridge2 = 0.78 + 0.03 * np.sin(X * 5 + 2)
    m = (Y > ridge2)[..., None]
    img = np.where(m, np.array([15, 28, 60], dtype=np.float32), img)

    bright = 0.6 + 0.4 * np.sin(t * 3 + LPH)
    img[LIY, LIX] = np.array([255, 210, 90], dtype=np.float32) * bright[:, None]

    center = 0.13 + 0.015 * np.sin(Y * 9 + t * 1.8) * (1 - Y)
    hw = 0.095 * np.clip((Y - 0.28) / 0.5, 0, 1) ** 0.9 + 0.008 * np.sin(Y * 40 + t * 3)
    tree = (np.abs(X - center) * AS < hw) & (Y > 0.28)
    tone = 8 + 12 * (0.5 + 0.5 * np.sin(Y * 50 + X * 30))
    tree_col = np.stack([tone, tone * 2 + 6, tone * 1.5 + 4], axis=-1)
    img = np.where(tree[..., None], tree_col, img)

    return img


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Starry Night - ASCII")
    font = pygame.font.SysFont("consolas,couriernew,dejavusansmono,monospace", 12, bold=True)
    clock = pygame.time.Clock()
    cache = {}
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

        img = paint(t)
        lum = (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114) / 255
        idx = np.clip((lum ** 0.8) * (N - 1) + jitter, 0, N - 1).astype(int)
        col = np.clip(img * 1.25 + 15, 0, 255)
        q = np.clip((col / 51).astype(int), 0, 5) * 51

        screen.fill((0, 0, 0))
        blits = []
        for r in range(ROWS):
            for c in range(COLS):
                i = idx[r, c]
                if i == 0:
                    continue
                key = (i, q[r, c, 0], q[r, c, 1], q[r, c, 2])
                s = cache.get(key)
                if s is None:
                    s = font.render(RAMP[i], True, (int(key[1]), int(key[2]), int(key[3])))
                    cache[key] = s
                blits.append((s, (c * CW, r * CH)))
        screen.blits(blits)
        pygame.display.flip()


if __name__ == "__main__":
    main()
import random

import cv2
import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator


# sRGB -> linear light
def to_lin(x):
    x = np.asarray(x, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# Linear light -> sRGB
def to_srgb(y):
    y = np.clip(y, 0, 1)
    return 255.0 * np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)


# Load a texture (grayscale) or overlay (RGBA) in portrait orientation
def load_texture(path, rgba=False):
    t = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED if rgba else cv2.IMREAD_GRAYSCALE)
    return cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE) if t.shape[1] > t.shape[0] else t


# Paper color: mean of pixels above the Otsu threshold
def paper_color(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img[gray > t].mean(axis=0)


# Texture region at the scale of one text line on an A4 page
def crop_texture(tex, h, w, lines=55, jitter=(0.7, 1.4)):
    H, W = tex.shape[:2]
    k = H / (h * lines * random.uniform(*jitter))
    ph, pw = max(1, round(h * k)), max(1, round(w * k))
    s = min(1.0, W / pw, H / ph)
    ph, pw = max(1, int(ph * s)), max(1, int(pw * s))
    y, x = random.randint(0, H - ph), random.randint(0, W - pw)
    return tex[y:y + ph, x:x + pw]


def fit(p, h, w):
    interp = cv2.INTER_AREA if p.shape[0] >= h else cv2.INTER_LINEAR
    return cv2.resize(p, (w, h), interpolation=interp).astype(np.float32)


def patch(tex, h, w):
    return fit(crop_texture(tex, h, w), h, w)


# Wear mask: dark deviations from the blurred local background
def wear_mask(tex, sigma_frac=25 / 2339):
    g = tex.astype(np.float32)
    H, W = g.shape
    s = cv2.resize(g, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(s, (0, 0), sigma_frac * s.shape[0]), (W, H))
    return (np.clip((bg - g) / np.maximum(bg, 1), 0, 1) * 255).astype(np.uint8)


# Print defects on a 3x grid: boundary noise (Kanungo & Zheng 2004), toner particles (Norris & Barney Smith 2004)
def print_defects(img, paper_tex):
    ink = random.triangular(0, 2, 2)
    edge = random.triangular(0, 1, 1)
    grain = random.triangular(0, 2, 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    light, dark = gray > t, gray <= t
    bg, fg = np.median(gray[light]), np.percentile(gray[dark], 10)
    coverage = np.clip((bg - gray.astype(np.float32)) / (bg - fg), 0, 1)

    _, _, stats, _ = cv2.connectedComponentsWithStats((coverage >= 0.5).astype(np.uint8), connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    ok = (hs >= 5) & (hs < h) & (stats[1:, cv2.CC_STAT_WIDTH] <= 3 * hs) & (stats[1:, cv2.CC_STAT_AREA] >= 8)
    scale = max(0.25, (np.median(hs[ok]) if ok.any() else 0.65 * h) / 28)

    up = 3
    min_sigma = 0.25 * up / 4
    size = (w * up, h * up)
    ideal = (cv2.resize(coverage, size, interpolation=cv2.INTER_CUBIC) >= 0.5).astype(np.uint8)

    inside = np.maximum(cv2.distanceTransform(ideal, cv2.DIST_L2, 5) - 0.5, 0) / up
    outside = np.maximum(cv2.distanceTransform(1 - ideal, cv2.DIST_L2, 5) - 0.5, 0) / up
    near = np.exp(-((inside + outside) / max(0.08, 0.6 * scale * edge)) ** 2)
    u = np.random.rand(*ideal.shape)
    shaped = ideal.copy()
    shaped[(ideal == 1) & (u < 0.8 * edge * near)] = 0
    shaped[(ideal == 0) & (u < 0.7 * edge * near)] = 1
    shaped = cv2.morphologyEx(shaped, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    paper = patch(paper_tex, h, w)
    fiber = paper - cv2.GaussianBlur(paper, (0, 0), max(1.0, 3 * scale))
    fiber = np.clip((fiber - fiber.mean()) / max(fiber.std(), 1e-6), -3, 3)
    fiber = cv2.resize(fiber, size, interpolation=cv2.INTER_LINEAR)
    r = max(1, round(0.18 * scale * up))
    particle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    density = -np.log(min(0.15, max(0.001, 0.045 * ink))) / particle.sum()
    exposure = cv2.GaussianBlur(shaped.astype(np.float32), (0, 0), max(min_sigma, 0.12 * scale * up))
    hit = 1 - np.exp(-density * exposure * np.exp(-0.16 * ink * fiber))
    printed = cv2.dilate((np.random.rand(*ideal.shape) < hit).astype(np.uint8), particle).astype(np.float32)

    change = cv2.GaussianBlur(ideal.astype(np.float32) - printed, (0, 0), max(min_sigma, 0.18 * scale * up))
    change = cv2.resize(change, (w, h), interpolation=cv2.INTER_AREA)
    src = img.astype(np.float32)
    span = np.maximum(np.median(src[light], axis=0) - np.percentile(src[dark], 10, axis=0), 0)
    out = np.clip(src + change[:, :, None] * span, 0, 255)
    out = to_srgb(to_lin(out) * np.maximum(to_lin(paper), 1e-6)[:, :, None] ** grain)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# Ink color from hue at full saturation
def ink_color(hue, v=181):
    hsv = np.uint8([[[round(hue / 2) % 180, 255, v]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


# Marks: RGBA overlay as transmission in linear light; black/colored ink is partly worn by a mask
def marks(img, overlay, black_overlay, wear):
    h, w = img.shape[:2]
    r = random.random()
    p = crop_texture(overlay if r < 0.8 else black_overlay, h, w)
    a = 1 - (1 - p[:, :, 3:4].astype(np.float32) / 255) ** random.uniform(1, 6)
    if r < 0.8:
        T = fit(1 - a * (1 - to_lin(p[:, :, :3])), h, w)
    else:
        c = np.zeros(3) if r < 0.9 else to_lin(ink_color(random.uniform(180, 330)))
        a = fit(a[:, :, 0], h, w)[:, :, None]
        a = a * (1 - patch(wear, h, w)[:, :, None] / 255 * random.random())
        T = 1 - a * (1 - c)
    return np.clip(to_srgb(to_lin(img) * T), 0, 255).astype(np.uint8)


# Erase: pull ink toward the paper color by a wear mask
def erase(img, wear):
    h, w = img.shape[:2]
    M = patch(wear, h, w)[:, :, None] / 255 * random.random()
    L, P = to_lin(img), to_lin(paper_color(img))
    return np.clip(to_srgb(L + (P - L) * M), 0, 255).astype(np.uint8)


# Paper: multiply by a texture with white point sampled between its median and 255
def paper_texture(img, tex):
    h, w = img.shape[:2]
    P = patch(tex, h, w)[:, :, None]
    P = np.clip(P * 255 / random.uniform(np.median(P), 255), 0, 255)
    return np.clip(to_srgb(to_lin(img) * to_lin(P)), 0, 255).astype(np.uint8)


# Page curl: power-law vertical bend with horizontal squeeze toward the edge
def curl(img):
    h, w = img.shape[:2]
    amp, length = random.uniform(0, 1), random.uniform(1, 13)
    power, squeeze = random.uniform(1, 3.5), random.uniform(0, 0.8)
    side, up = random.choice(("right", "left", "both")), random.random() < 0.5
    L = length * h
    px = np.arange(w) + 0.5
    d = px - (w - L) if side == "right" else (L - px if side == "left" else np.abs(px - w / 2) - (w / 2 - L))
    u = np.clip(d / L, 0, 1)
    dy = amp * h * u ** power
    X = np.concatenate([[0], np.cumsum(1 - squeeze * u)])
    W2, H2 = max(2, int(np.ceil(X[-1]))), h + int(np.ceil(dy.max()))
    src_x = np.interp(np.arange(W2) + 0.5, X, np.arange(w + 1)) - 0.5
    dy_new = np.interp(src_x, np.arange(w), dy)
    Y = np.arange(H2, dtype=np.float32)[:, None]
    map_y = (Y - (H2 - h) + dy_new[None, :] if up else Y - dy_new[None, :]).astype(np.float32)
    map_x = np.broadcast_to(src_x[None, :], (H2, W2)).astype(np.float32)
    bg = tuple(float(c) for c in paper_color(img))
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


# Uniform linear motion blur kernel
def motion_kernel(length, angle):
    radius = int(np.ceil(length / 2)) + 1
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.float32)
    t = np.linspace(-length / 2, length / 2, max(3, int(np.ceil(length * 8)) + 1))
    a = np.deg2rad(angle)
    x, y = radius + t * np.cos(a), radius + t * np.sin(a)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy
    weights = np.ones(len(t), np.float32)
    weights[[0, -1]] = 0.5
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            np.add.at(kernel, (iy + dy, ix + dx), weights * wx * wy)
    return kernel / kernel.sum()


# Scan: Gaussian blur, motion blur, contrast around mid-gray
def scan(img):
    sigma = random.uniform(0, 1)
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    length = random.uniform(0, 6)
    if length > 0:
        img = cv2.filter2D(img, -1, motion_kernel(length, random.uniform(0, 180)), borderType=cv2.BORDER_REPLICATE)
    contrast = random.uniform(1, 1.5)
    return np.clip((img.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)


# Low resolution: downsample and resize back
def lowres(img, min_h=13):
    h, w = img.shape[:2]
    nh = random.uniform(min_h, h)
    small = cv2.resize(img, (max(1, round(w * nh / h)), round(nh)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# Gaussian intensity noise shared across channels
def noise(img):
    grain = np.random.normal(0, random.uniform(1, 5), img.shape[:2] + (1,))
    return np.clip(np.rint(img + grain), 0, 255).astype(np.uint8)


# JPEG compression artifacts
def jpeg(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(1, 70)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def smooth_noise(h, w, cells):
    n = np.random.standard_normal((cells, max(2, round(cells * w / h)))).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / max(n.std(), 1e-6)


# Paper surface height: smooth bend along a random axis, a curled corner, low wrinkles
def height_field(U, V, W, H):
    th = random.uniform(0, np.pi)
    s = (U - W / 2) * np.cos(th) + (V - H / 2) * np.sin(th)
    k = random.randint(3, 5)
    curve = CubicSpline(np.linspace(0, 1, k), np.random.uniform(-1, 1, k), bc_type="natural")
    z = random.uniform(0.02, 0.07) * H * curve((s - s.min()) / np.ptp(s))

    cx, cy = random.choice([(0, 0), (W, 0), (0, H), (W, H)])
    d = np.array([W / 2 - cx, H / 2 - cy]) / np.hypot(W / 2, H / 2)
    L = random.uniform(0.15, 0.45) * min(W, H)
    c = np.clip(1 - ((U - cx) * d[0] + (V - cy) * d[1]) / L, 0, 1)
    z += random.uniform(0.15, 0.5) * L * c ** 2

    return z + random.uniform(0.002, 0.006) * H * smooth_noise(*U.shape, 6)


def rotation(ax, ay, az):
    ax, ay, az = np.deg2rad([ax, ay, az])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# Background: base color with anisotropic streaks (wood/fabric), coarse unevenness and grain
def background(h, w):
    hsv = np.uint8([[[random.uniform(0, 30), random.uniform(15, 140), random.uniform(60, 160)]]])
    base = to_lin(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    k = h / 2339
    streak = cv2.GaussianBlur(np.random.standard_normal((h, w)).astype(np.float32), (0, 0),
                              sigmaX=random.uniform(10, 60) * k, sigmaY=random.uniform(0.8, 3) * k)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-90, 90), 1.0)
    streak = cv2.warpAffine(streak, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    streak /= max(streak.std(), 1e-6)
    grain = cv2.GaussianBlur(np.random.standard_normal((h, w)).astype(np.float32), (0, 0), 0.8)
    grain /= max(grain.std(), 1e-6)
    tex = (1 + random.uniform(0.04, 0.15) * streak + random.uniform(0.04, 0.1) * smooth_noise(h, w, 5)
           + random.uniform(0.02, 0.05) * grain)
    return base[None, None, :] * np.clip(tex, 0.3, 2)[:, :, None]


# Depth of field: per-pixel blur sigma interpolated over a stack of Gaussian blurs
def depth_blur(img, sigma):
    levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5.5, 7, 9, 12], np.float32)
    levels = levels[: np.searchsorted(levels, sigma.max()) + 1]
    stack = [img if s == 0 else cv2.GaussianBlur(img, (0, 0), float(s)) for s in levels]
    idx = np.clip(np.searchsorted(levels, sigma, side="right") - 1, 0, len(levels) - 2)
    f = np.clip((sigma - levels[idx]) / (levels[idx + 1] - levels[idx]), 0, 1)[:, :, None]
    out = np.zeros_like(img)
    for i in range(len(levels) - 1):
        out += (idx == i)[:, :, None] * (stack[i] * (1 - f) + stack[i + 1] * f)
    return out


# Page photo: bent paper under a perspective camera, Lambert shading, background, shadow, depth-of-field blur
def page_photo(img, paper, grid=(90, 64)):
    H, W = img.shape[:2]
    U, V = np.meshgrid(np.linspace(0, W - 1, grid[1]), np.linspace(0, H - 1, grid[0]))
    z = height_field(U, V, W, H)

    R = rotation(random.uniform(-22, 22), random.uniform(-18, 18), random.uniform(-6, 6))
    P = np.stack([U - W / 2, V - H / 2, -z], -1) @ R.T
    P0 = np.stack([U - W / 2, V - H / 2, np.zeros_like(z)], -1) @ R.T
    D = random.uniform(1.6, 3.2) * H
    Zc, Zc0 = P[..., 2] + D, P0[..., 2] + D
    x, y = D * P[..., 0] / Zc, D * P[..., 1] / Zc
    x0, y0 = D * P0[..., 0] / Zc0, D * P0[..., 1] / Zc0

    sc = random.uniform(0.78, 0.92) * min(W / np.ptp(x), H / np.ptp(y))
    ox, oy = random.uniform(0, W - np.ptp(x) * sc), random.uniform(0, H - np.ptp(y) * sc)
    xmin, ymin = x.min(), y.min()
    x, y = (x - xmin) * sc + ox, (y - ymin) * sc + oy
    x0, y0 = (x0 - xmin) * sc + ox, (y0 - ymin) * sc + oy

    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n *= -np.sign(n[..., 2:3])
    light = np.array([random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7), -1.0])
    light /= np.linalg.norm(light)
    amb = random.uniform(0.35, 0.6)
    shade = amb + (1 - amb) * np.clip(n @ light, 0, 1)
    shade /= np.percentile(shade, 95)

    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = LinearNDInterpolator(np.stack([x.ravel(), y.ravel()], 1),
                             np.stack([U.ravel(), V.ravel(), 1 / Zc.ravel(), shade.ravel()], 1))(gx, gy).astype(np.float32)
    xy = np.stack([x, y], -1)
    outline = np.concatenate([xy[0], xy[1:, -1], xy[-1, ::-1], xy[-2:0:-1, 0]])
    alpha = np.zeros((H, W), np.uint8)
    cv2.fillPoly(alpha, [np.round(outline * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    alpha = alpha.astype(np.float32)[:, :, None] / 255
    inside = ~np.isnan(m[..., 0]) & (alpha[..., 0] > 0.5)
    m[np.isnan(m)] = 0
    map_u, map_v, invz_page, shade_px = (m[..., i] for i in range(4))

    Pt = cv2.resize(paper, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    Pt = np.clip(Pt * 255 / random.uniform(np.median(Pt), 255), 0, 255)
    page = to_lin(img) * to_lin(Pt)[:, :, None] * np.array([random.uniform(0.93, 1), random.uniform(0.95, 1), 1])
    page = cv2.remap(page.astype(np.float32), map_u, map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    page *= np.clip(shade_px, 0, 1.2)[:, :, None] * random.uniform(0.85, 1.0)

    bg = background(H, W)
    off = -light[:2] / abs(light[2]) * random.uniform(0.004, 0.015) * H
    shadow = cv2.warpAffine(inside.astype(np.float32), np.float32([[1, 0, off[0]], [0, 1, off[1]]]), (W, H))
    shadow = cv2.GaussianBlur(shadow, (0, 0), random.uniform(0.004, 0.015) * H)
    bg *= (1 - random.uniform(0.35, 0.65) * shadow)[:, :, None]

    out = bg * (1 - alpha) + page * alpha

    th = random.uniform(0, 2 * np.pi)
    ramp = ((gx - W / 2) * np.cos(th) + (gy - H / 2) * np.sin(th)) / max(W, H)
    r2 = ((gx - W / 2) / W) ** 2 + ((gy - H / 2) / H) ** 2
    out *= ((1 + random.uniform(0.1, 0.35) * ramp) * (1 - random.uniform(0.15, 0.45) * r2))[:, :, None]

    coef = np.linalg.lstsq(np.stack([x0.ravel(), y0.ravel(), np.ones(x0.size)], 1), 1 / Zc0.ravel(), rcond=None)[0]
    invz = np.where(inside, invz_page, coef[0] * gx + coef[1] * gy + coef[2])
    focus = 1 / Zc.flat[random.randrange(Zc.size)]
    smax = random.uniform(0.5, 2) * H / 2339
    sigma = np.clip(smax * np.abs(invz - focus) / np.abs(1 / Zc - focus).max(), 0, 1.6 * smax)
    out = depth_blur(out.astype(np.float32), sigma.astype(np.float32))

    out = to_srgb(out) + np.random.normal(0, random.uniform(1, 3), (H, W, 1))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

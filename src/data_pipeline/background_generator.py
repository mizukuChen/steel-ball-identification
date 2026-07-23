"""程序化生成噪音/纹理背景图，用于合成数据的背景多样性补充。"""
import numpy as np


def _random_color() -> np.ndarray:
    return np.random.randint(0, 256, size=3).astype(np.uint8)


def gen_solid(h: int, w: int) -> np.ndarray:
    """纯色背景。"""
    color = _random_color()
    return np.tile(color, (h, w, 1))


def gen_gaussian_noise(h: int, w: int) -> np.ndarray:
    """高斯噪音背景。"""
    base = _random_color().astype(np.float32)
    noise = np.random.normal(0, np.random.uniform(20, 60), size=(h, w, 3))
    img = base[None, None, :] + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_gradient(h: int, w: int) -> np.ndarray:
    """随机方向的线性渐变背景。"""
    c1 = _random_color().astype(np.float32)
    c2 = _random_color().astype(np.float32)
    angle = np.random.uniform(0, 2 * np.pi)
    xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    t = xs * np.cos(angle) + ys * np.sin(angle)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    img = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_cloud_texture(h: int, w: int) -> np.ndarray:
    """模拟云状/斑驳纹理（多层低频噪音叠加，近似柏林噪音效果）。"""
    import cv2

    base_color = _random_color().astype(np.float32)
    texture = np.zeros((h, w), dtype=np.float32)

    for scale in [4, 8, 16, 32]:
        small = np.random.uniform(-1, 1, size=(scale, scale)).astype(np.float32)
        layer = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        texture += layer / scale

    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
    intensity = np.random.uniform(30, 80)
    img = base_color[None, None, :] + (texture[..., None] - 0.5) * intensity
    return np.clip(img, 0, 255).astype(np.uint8)


def gen_stripes(h: int, w: int) -> np.ndarray:
    """随机条纹背景。"""
    import cv2

    c1 = _random_color()
    c2 = _random_color()
    stripe_width = np.random.randint(10, 60)
    angle = np.random.uniform(0, 180)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    proj = xs * np.cos(np.radians(angle)) + ys * np.sin(np.radians(angle))
    mask = ((proj // stripe_width) % 2 == 0)

    img = np.where(mask[..., None], c1, c2).astype(np.uint8)
    if np.random.rand() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=np.random.uniform(0.5, 3))
    return img


def gen_checkerboard(h: int, w: int) -> np.ndarray:
    """随机棋盘格背景。"""
    c1 = _random_color()
    c2 = _random_color()
    cell = np.random.randint(15, 80)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    mask = ((xs // cell + ys // cell) % 2 == 0)
    return np.where(mask[..., None], c1, c2).astype(np.uint8)


GENERATORS = [
    gen_solid,
    gen_gaussian_noise,
    gen_gradient,
    gen_cloud_texture,
    gen_stripes,
    gen_checkerboard,
]


def generate_random_background(h: int = 600, w: int = 800) -> np.ndarray:
    """随机选择一种生成方式，产生一张背景图（BGR）。"""
    generator = GENERATORS[np.random.randint(len(GENERATORS))]
    return generator(h, w)

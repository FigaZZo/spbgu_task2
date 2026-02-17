import cv2
import numpy as np


# -------------------------
# 1) Маска листвы (сильнее и аккуратнее)
# -------------------------

def build_coarse_foliage_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Грубая (перекрывающая) маска листвы:
    - ловим зелёные + жёлто-оранжевые оттенки
    - отсекаем небо по низкой насыщенности и высокой яркости
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Листва: достаточно насыщенная и не слишком тёмная
    base = (S > 30) & (V > 30)

    # Зелёная зона (лето)
    green = (H >= 28) & (H <= 95)
    # Жёлто-оранжевая зона (осень)
    autumn = (H >= 5) & (H < 35)

    foliage = base & (green | autumn)

    # Небо/облака: низкая насыщенность + высокая яркость
    sky = (S < 55) & (V > 170)

    mask = np.where(foliage & (~sky), 255, 0).astype(np.uint8)

    # Морфология, чтобы заполнить мелкие пробелы
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def fill_holes(mask_255: np.ndarray) -> np.ndarray:
    """Заполняем дыры в бинарной маске (0/255) через flood fill"""
    h, w = mask_255.shape[:2]
    mask = mask_255.copy()
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    # flood fill от угла (считаем фон)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    inv_flood = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, inv_flood)
    return filled


def refine_foliage_mask(img_bgr: np.ndarray, iters: int = 5) -> np.ndarray:
    """
    1) Делаем грубую маску
    2) Инициализируем GrabCut
    3) После GrabCut: заполняем дырки, чуть расширяем
    """
    coarse = build_coarse_foliage_mask(img_bgr)
    h, w = coarse.shape[:2]

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Точно фон: небо (уже исключено), и явно не-листва
    gc_mask[coarse == 0] = cv2.GC_PR_BGD
    # Вероятная листва
    gc_mask[coarse > 0] = cv2.GC_PR_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)

    foliage = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Заполняем дырки + сглаживаем
    foliage = fill_holes(foliage)

    kernel = np.ones((9, 9), np.uint8)
    foliage = cv2.morphologyEx(foliage, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Лёгкое расширение, чтобы не оставались "пропуски" вокруг веток/стволов
    foliage = cv2.dilate(foliage, np.ones((5, 5), np.uint8), iterations=1)

    return foliage


# -------------------------
# 2) "Умный" целевой цвет из референса (убирает кислотность)
# -------------------------

def foliage_hsv_stats(img_bgr: np.ndarray, mask_255: np.ndarray):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = mask_255 > 0
    H = hsv[:, :, 0][m].astype(np.float32)
    S = hsv[:, :, 1][m].astype(np.float32)
    V = hsv[:, :, 2][m].astype(np.float32)

    if H.size < 200:
        # запасной вариант
        return 60.0, 120.0, 150.0

    # Медианы устойчивее к выбросам
    return float(np.median(H)), float(np.median(S)), float(np.median(V))


def pull_hue_circular(H, target_H, alpha):
    """
    Плавно тянем Hue по кругу 0..179 по кратчайшему пути.
    """
    d = ((target_H - H + 90) % 180) - 90
    return (H + d * alpha) % 180


def recolor_by_reference(
    src_bgr: np.ndarray,
    src_mask_255: np.ndarray,
    ref_bgr: np.ndarray,
    ref_mask_255: np.ndarray,
    strength: float = 0.75,
    sat_match: float = 0.60,
    val_match: float = 0.15,
) -> np.ndarray:
    """
    Вместо фиксированного Hue: берём целевой Hue/S/V из референса листвы и тянем к нему
    - strength: сила изменения Hue (0..1). 0.75 обычно выглядит естественно.
    - sat_match: наскоько подтягивать насыщенность к референсу
    - val_match: насколько подтягивать яркость к референсу
    """
    hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    m = src_mask_255 > 0
    if m.sum() < 200:
        return src_bgr.copy()

    tgt_H, tgt_S, tgt_V = foliage_hsv_stats(ref_bgr, ref_mask_255)

    # мягкая альфа по маске
    alpha = cv2.GaussianBlur(src_mask_255, (31, 31), 0).astype(np.float32) / 255.0
    alpha = alpha * strength

    H_new = pull_hue_circular(H, tgt_H, alpha)

    # Насыщенность: не умножаем "в потолок", а тянем к целевой медиане
    S_new = S + (tgt_S - S) * (alpha * sat_match)
    # Яркость: очень осторожно
    V_new = V + (tgt_V - V) * (alpha * val_match)

    hsv[:, :, 0] = H_new
    hsv[:, :, 1] = np.clip(S_new, 0, 255)
    hsv[:, :, 2] = np.clip(V_new, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out


# -------------------------
# 3) Основной пайплайн
# -------------------------

def process(photo_from: str, photo_ref: str, out_path: str):
    src = cv2.imread(photo_from)
    ref = cv2.imread(photo_ref)
    if src is None or ref is None:
        raise FileNotFoundError("Не удалось прочитать входные изображения.")

    src_mask = refine_foliage_mask(src, iters=5)
    ref_mask = refine_foliage_mask(ref, iters=5)

    out = recolor_by_reference(
        src, src_mask,
        ref, ref_mask,
        strength=0.75,      # меньше неона
        sat_match=0.60,     # тянем насыщенность к референсу, а не усиливаем
        val_match=0.12      # яркость почти не трогаем
    )

    cv2.imwrite(out_path, out)


if __name__ == "__main__":
    # Photo1 (осень) -> Summer.jpg, референс лето = Photo2
    process("Photo1.jpg", "Photo2.jpg", "Summer.jpg")

    # Photo2 (лето) -> Autumn.jpg, референс осень = Photo1
    process("Photo2.jpg", "Photo1.jpg", "Autumn.jpg")

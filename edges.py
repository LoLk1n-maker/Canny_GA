"""
edges.py — работа с изображением и оценка качества контуров.

Здесь собрано всё, что относится к обработке изображения:
загрузка, перевод в оттенки серого, вызов алгоритма Кэнни и метрики,
по которым оценивается качество полученной карты контуров.
Генетический алгоритм (ga.py) использует функцию quality() как
функцию приспособленности и ничего не знает про OpenCV.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Ядро для подсчёта числа соседей у каждого пикселя (центр = 0)
_NEIGHBOURS = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.float32)


def load_image(path: str) -> np.ndarray:
    """Загружает изображение в BGR.

    cv2.imread() не умеет открывать пути с кириллицей в имени файла
    (на Windows он молча возвращает None), поэтому читаем файл сами
    и декодируем из памяти.
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Файл пуст или недоступен: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось распознать изображение: {path}")
    return img


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Перевод в оттенки серого — вход алгоритма Кэнни."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def canny(gray: np.ndarray, t1: int, t2: int) -> np.ndarray:
    """Выделение контуров алгоритмом Кэнни с порогами t1 (нижний) и t2 (верхний)."""
    return cv2.Canny(gray, int(t1), int(t2))


@dataclass
class EdgeMetrics:
    """Числовые характеристики карты контуров."""
    edge_ratio: float      # доля контурных пикселей от всех пикселей
    isolated_ratio: float  # доля одиночных («шумовых») контурных пикселей
    continuity: float      # доля контурных пикселей, имеющих 2+ соседей
    score: float = 0.0     # итоговая оценка качества (чем больше, тем лучше)

    def as_text(self) -> str:
        return (f"контурных пикселей {self.edge_ratio * 100:.2f}%, "
                f"непрерывность {self.continuity * 100:.1f}%, "
                f"шум {self.isolated_ratio * 100:.2f}%")


def measure(edges: np.ndarray) -> EdgeMetrics:
    """Считает характеристики карты контуров.

    Контур хорошего качества — это длинные связные линии, а не редкая
    крапина из отдельных точек. Поэтому кроме доли контурных пикселей
    смотрим на связность: сколько соседей у каждого контурного пикселя.
    """
    mask = (edges > 0).astype(np.float32)
    edge_count = float(mask.sum())
    total = float(mask.size)

    if edge_count == 0:
        return EdgeMetrics(edge_ratio=0.0, isolated_ratio=0.0, continuity=0.0)

    neighbours = cv2.filter2D(mask, -1, _NEIGHBOURS, borderType=cv2.BORDER_CONSTANT)
    isolated = float(((mask > 0) & (neighbours == 0)).sum())
    linked = float(((mask > 0) & (neighbours >= 2)).sum())

    return EdgeMetrics(
        edge_ratio=edge_count / total,
        isolated_ratio=isolated / edge_count,
        continuity=linked / edge_count,
    )


def quality(edges: np.ndarray, target_ratio: float = 0.10,
            w_ratio: float = 10.0, w_noise: float = 1.0,
            w_continuity: float = 0.5) -> EdgeMetrics:
    """Функция приспособленности: оценка карты контуров одним числом.

    Складывается из трёх составляющих:
      1) отклонение доли контурных пикселей от целевой (главный член);
      2) штраф за одиночные пиксели — это шум;
      3) премия за непрерывность линий.

    В исходной версии программы учитывался только п. 1, из-за чего
    алгоритм мог выбрать пороги, дающие нужное количество пикселей,
    но рассыпанных по всему кадру.
    """
    m = measure(edges)
    m.score = (-w_ratio * abs(m.edge_ratio - target_ratio)
               - w_noise * m.isolated_ratio
               + w_continuity * m.continuity)
    return m

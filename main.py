"""
main.py — точка входа.

Два режима работы:
    python main.py                      — графический интерфейс (Tkinter)
    python main.py path/to/image.jpg    — расчёт в консоли, без окна

Консольный режим удобен для отчёта: он печатает найденные пороги,
метрики качества и сравнение с фиксированными порогами (100, 200),
а также сохраняет обе карты контуров рядом с исходным файлом.
"""
import argparse
import os
import sys

import cv2

import edges as E
from ga import GAConfig, evolve

DEFAULT_T1, DEFAULT_T2 = 100, 200   # «обычный» Кэнни для сравнения


def run_console(path: str, cfg: GAConfig, save: bool = True) -> int:
    """Расчёт без графического интерфейса."""
    try:
        img = E.load_image(path)
    except (OSError, ValueError) as err:
        print("Ошибка:", err)
        return 1

    gray = E.to_gray(img)
    print(f"Изображение: {os.path.basename(path)}  {gray.shape[1]}x{gray.shape[0]} px")
    print(f"Популяция {cfg.population_size}, поколений {cfg.generations}, "
          f"целевая доля контуров {cfg.target_ratio * 100:.0f}%\n")

    def show(gen, total, best, score):
        if gen == 1 or gen == total or gen % 5 == 0:
            print(f"  поколение {gen:>3}/{total}: лучшие пороги "
                  f"({best[0]:>3}, {best[1]:>3}), оценка {score:+.4f}")

    result = evolve(gray, cfg, progress=show)

    edges_ga = E.canny(gray, result.t1, result.t2)
    edges_def = E.canny(gray, DEFAULT_T1, DEFAULT_T2)
    m_def = E.quality(edges_def, cfg.target_ratio)

    print("\n" + "=" * 62)
    print(f"  Кэнни без ГА  ({DEFAULT_T1}, {DEFAULT_T2}): {m_def.as_text()}")
    print(f"  Кэнни + ГА    ({result.t1}, {result.t2}): {result.metrics.as_text()}")
    print("-" * 62)
    print(f"  оценка качества: без ГА {m_def.score:+.4f}   "
          f"с ГА {result.metrics.score:+.4f}")
    print(f"  вычислений Кэнни: {result.evaluations} "
          f"(без кэширования было бы ~{cfg.population_size * cfg.generations})")
    print("=" * 62)

    if save:
        stem, _ = os.path.splitext(path)
        p1, p2 = stem + "_canny_default.png", stem + "_canny_ga.png"
        cv2.imwrite(p1, edges_def)
        cv2.imwrite(p2, edges_ga)
        print(f"Сохранено: {os.path.basename(p1)}, {os.path.basename(p2)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Выделение контуров алгоритмом Кэнни с подбором порогов "
                    "генетическим алгоритмом.")
    parser.add_argument("image", nargs="?", help="путь к изображению (без него — окно программы)")
    parser.add_argument("-p", "--population", type=int, default=20, help="размер популяции")
    parser.add_argument("-g", "--generations", type=int, default=30, help="число поколений")
    parser.add_argument("-t", "--target", type=float, default=10.0,
                        help="целевая доля контурных пикселей, %%")
    parser.add_argument("--seed", type=int, default=None, help="зерно случайных чисел")
    parser.add_argument("--no-save", action="store_true", help="не сохранять результат в файлы")
    args = parser.parse_args(argv)

    cfg = GAConfig(population_size=args.population,
                   generations=args.generations,
                   target_ratio=args.target / 100.0,
                   seed=args.seed)

    if args.image:
        return run_console(args.image, cfg, save=not args.no_save)

    try:
        from gui import run_gui
    except ImportError as err:
        print("Не удалось запустить графический интерфейс:", err)
        print("Укажите путь к изображению для работы в консоли: python main.py image.jpg")
        return 1
    run_gui(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

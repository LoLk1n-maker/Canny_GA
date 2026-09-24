"""
ga.py — генетический алгоритм подбора порогов Кэнни.

Особь (индивид) — пара порогов (t1, t2), 0 <= t1 < t2 <= 255.
Приспособленность считает edges.quality(): она оценивает карту
контуров, полученную с этими порогами.

Отличия от простейшей реализации:
  * особи создаются сразу корректными (t1 < t2), а не отбраковываются;
  * отбор турнирный, а не «топ-5» — сохраняется разнообразие популяции;
  * скрещивание смешивающее (BLX-α), а не усреднение: усреднение
    быстро схлопывает популяцию в одну точку;
  * сила мутации убывает с номером поколения (сначала исследуем, потом уточняем);
  * значения приспособленности кэшируются — Кэнни не считается дважды
    для одной и той же пары порогов.
"""
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from edges import canny, quality, EdgeMetrics

Individual = Tuple[int, int]

T_MIN, T_MAX = 0, 255
MIN_GAP = 10          # минимальный зазор между нижним и верхним порогом


@dataclass
class GAConfig:
    """Параметры работы генетического алгоритма."""
    population_size: int = 20
    generations: int = 30
    target_ratio: float = 0.10   # желаемая доля контурных пикселей
    mutation_rate: float = 0.3   # вероятность мутации потомка
    elite: int = 2               # сколько лучших особей переходит без изменений
    tournament: int = 3          # размер турнира при отборе
    seed: Optional[int] = None   # фиксируется для воспроизводимости


@dataclass
class GAResult:
    """Результат работы алгоритма."""
    best: Individual
    metrics: EdgeMetrics
    history: List[Tuple[int, float, float]] = field(default_factory=list)
    evaluations: int = 0         # сколько раз реально вызывался Кэнни

    @property
    def t1(self) -> int:
        return self.best[0]

    @property
    def t2(self) -> int:
        return self.best[1]


def _repair(t1: int, t2: int) -> Individual:
    """Приводит пару порогов к допустимому виду: 0 <= t1 < t2 <= 255."""
    t1 = int(max(T_MIN, min(T_MAX - MIN_GAP, t1)))
    t2 = int(max(t1 + MIN_GAP, min(T_MAX, t2)))
    return t1, t2


def random_individual(rng: random.Random) -> Individual:
    """Случайная особь. Сразу корректная, поэтому «мёртвых» особей нет."""
    t1 = rng.randint(T_MIN, 150)
    return _repair(t1, t1 + rng.randint(MIN_GAP, 150))


def crossover(a: Individual, b: Individual, rng: random.Random) -> Individual:
    """Смешивающее скрещивание BLX-α: потомок берётся из интервала между
    родителями с небольшим выходом за его границы."""
    alpha = 0.3
    child = []
    for ga_, gb in zip(a, b):
        lo, hi = min(ga_, gb), max(ga_, gb)
        d = hi - lo
        child.append(rng.uniform(lo - alpha * d, hi + alpha * d))
    return _repair(round(child[0]), round(child[1]))


def mutate(ind: Individual, rng: random.Random, strength: float) -> Individual:
    """Мутация: сдвиг порогов на случайную величину.
    strength — от 1.0 в начале до ~0.2 в конце работы алгоритма."""
    span = max(4, int(30 * strength))
    return _repair(ind[0] + rng.randint(-span, span),
                   ind[1] + rng.randint(-span, span))


def _tournament(population: List[Individual], scores: dict,
                size: int, rng: random.Random) -> Individual:
    """Турнирный отбор: из случайной группы особей побеждает лучшая."""
    group = rng.sample(population, min(size, len(population)))
    return max(group, key=lambda ind: scores[ind])


def evolve(gray: np.ndarray, cfg: GAConfig = GAConfig(),
           progress: Optional[Callable[[int, int, Individual, float], None]] = None
           ) -> GAResult:
    """Основной цикл генетического алгоритма.

    progress — необязательный колбэк (поколение, всего, лучшая особь, оценка):
    через него графический интерфейс показывает ход поиска.
    """
    rng = random.Random(cfg.seed)
    cache: dict = {}

    def score_of(ind: Individual) -> float:
        """Приспособленность с кэшированием: одна пара порогов — один Кэнни."""
        if ind not in cache:
            cache[ind] = quality(canny(gray, *ind), cfg.target_ratio)
        return cache[ind].score

    population = [random_individual(rng) for _ in range(cfg.population_size)]
    history: List[Tuple[int, float, float]] = []

    for gen in range(1, cfg.generations + 1):
        scores = {ind: score_of(ind) for ind in set(population)}
        population.sort(key=lambda ind: scores[ind], reverse=True)

        best = population[0]
        mean = sum(scores[ind] for ind in population) / len(population)
        history.append((gen, scores[best], mean))
        if progress is not None:
            progress(gen, cfg.generations, best, scores[best])

        if gen == cfg.generations:
            break

        # элита переходит в новое поколение без изменений
        new_population = population[:max(1, cfg.elite)]
        strength = 1.0 - 0.8 * (gen / cfg.generations)   # мутация слабеет к концу
        while len(new_population) < cfg.population_size:
            p1 = _tournament(population, scores, cfg.tournament, rng)
            p2 = _tournament(population, scores, cfg.tournament, rng)
            child = crossover(p1, p2, rng)
            if rng.random() < cfg.mutation_rate:
                child = mutate(child, rng, strength)
            new_population.append(child)
        population = new_population

    best = max(cache, key=lambda ind: cache[ind].score)
    return GAResult(best=best, metrics=cache[best],
                    history=history, evaluations=len(cache))

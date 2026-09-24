"""
gui.py — графический интерфейс (Tkinter).

Показывает три изображения: оригинал, результат Кэнни с фиксированными
порогами и результат Кэнни с порогами, подобранными генетическим алгоритмом.
Под каждым выводятся метрики качества, поэтому сравнение не только
визуальное, но и числовое.

Поиск выполняется в отдельном потоке, поэтому окно не «замерзает»,
а в строке состояния видно текущее поколение.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

import edges as E
from ga import GAConfig, evolve

PREVIEW = 320          # размер области под картинку, px
DEFAULT_T1, DEFAULT_T2 = 100, 200


def _to_preview(array: np.ndarray, box: int = PREVIEW) -> ImageTk.PhotoImage:
    """Готовит изображение для показа, сохраняя пропорции."""
    if array.ndim == 3:
        array = array[:, :, ::-1]          # BGR -> RGB
    img = Image.fromarray(array)
    img.thumbnail((box, box), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


class App:
    def __init__(self, root: tk.Tk, cfg: GAConfig):
        self.root = root
        self.cfg = cfg
        self.gray = None
        self.image_path = None
        self.edges_default = None
        self.edges_ga = None
        self._queue: queue.Queue = queue.Queue()
        self._previews = {}                # ссылки на PhotoImage, иначе Tk их удалит

        root.title("Кэнни + генетический алгоритм — подбор порогов")

        # --- Три изображения с подписями ---
        panel = tk.Frame(root, padx=8, pady=8)
        panel.pack()
        self.labels, self.captions, self.metrics = {}, {}, {}
        for col, (key, title) in enumerate([("original", "Оригинал"),
                                            ("default", f"Кэнни без ГА ({DEFAULT_T1}, {DEFAULT_T2})"),
                                            ("ga", "Кэнни + ГА")]):
            holder = tk.Frame(panel, width=PREVIEW, height=PREVIEW,
                              relief=tk.SUNKEN, borderwidth=1)
            holder.grid(row=0, column=col, padx=6)
            holder.pack_propagate(False)
            lbl = tk.Label(holder, bg="#f0f0f0")
            lbl.pack(expand=True)
            self.labels[key] = lbl

            cap = tk.Label(panel, text=title, font=("Segoe UI", 10, "bold"))
            cap.grid(row=1, column=col, pady=(6, 0))
            self.captions[key] = cap

            met = tk.Label(panel, text="—", font=("Segoe UI", 8), fg="#555",
                           wraplength=PREVIEW, justify=tk.CENTER)
            met.grid(row=2, column=col)
            self.metrics[key] = met

        # --- Параметры алгоритма ---
        controls = tk.LabelFrame(root, text="Параметры генетического алгоритма",
                                 padx=8, pady=6)
        controls.pack(fill=tk.X, padx=14, pady=4)

        self.pop_var = tk.StringVar(value=str(cfg.population_size))
        self.gen_var = tk.StringVar(value=str(cfg.generations))
        self.target_var = tk.StringVar(value=f"{cfg.target_ratio * 100:.0f}")

        for col, (text, var, hint) in enumerate([
            ("Популяция", self.pop_var, "4–200"),
            ("Поколений", self.gen_var, "1–200"),
            ("Целевая доля контуров, %", self.target_var, "1–40"),
        ]):
            tk.Label(controls, text=text).grid(row=0, column=col * 3, sticky=tk.E, padx=(10, 2))
            tk.Entry(controls, textvariable=var, width=6, justify=tk.CENTER)\
                .grid(row=0, column=col * 3 + 1)
            tk.Label(controls, text=hint, fg="#888").grid(row=0, column=col * 3 + 2, sticky=tk.W)

        # --- Кнопки и строка состояния ---
        buttons = tk.Frame(root, pady=6)
        buttons.pack()
        self.btn_open = tk.Button(buttons, text="Открыть изображение…", width=22,
                                  command=self.open_image)
        self.btn_open.grid(row=0, column=0, padx=4)
        self.btn_run = tk.Button(buttons, text="Подобрать пороги", width=18,
                                 state=tk.DISABLED, command=self.start_search)
        self.btn_run.grid(row=0, column=1, padx=4)
        self.btn_save = tk.Button(buttons, text="Сохранить результат", width=20,
                                  state=tk.DISABLED, command=self.save_result)
        self.btn_save.grid(row=0, column=2, padx=4)

        self.status = tk.Label(root, text="Откройте изображение", anchor=tk.W,
                               relief=tk.SUNKEN, padx=6)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------ #

    def _set_preview(self, key: str, array: np.ndarray):
        photo = _to_preview(array)
        self._previews[key] = photo          # держим ссылку
        self.labels[key].config(image=photo)

    def _read_config(self) -> GAConfig:
        """Читает параметры из полей ввода с проверкой значений."""
        try:
            pop = int(self.pop_var.get())
            gen = int(self.gen_var.get())
            target = float(self.target_var.get().replace(",", "."))
        except ValueError:
            raise ValueError("Параметры должны быть числами.")
        if not 4 <= pop <= 200:
            raise ValueError("Размер популяции — от 4 до 200.")
        if not 1 <= gen <= 200:
            raise ValueError("Число поколений — от 1 до 200.")
        if not 1 <= target <= 40:
            raise ValueError("Целевая доля контуров — от 1 до 40 %.")
        return GAConfig(population_size=pop, generations=gen,
                        target_ratio=target / 100.0, seed=self.cfg.seed)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            img = E.load_image(path)
        except (OSError, ValueError) as err:
            messagebox.showerror("Не удалось открыть файл", str(err))
            return

        self.image_path = path
        self.gray = E.to_gray(img)
        self.edges_ga = None

        self._set_preview("original", img)
        self.edges_default = E.canny(self.gray, DEFAULT_T1, DEFAULT_T2)
        self._set_preview("default", self.edges_default)
        m = E.quality(self.edges_default)
        self.metrics["default"].config(text=m.as_text())
        self.metrics["ga"].config(text="—")
        self.captions["ga"].config(text="Кэнни + ГА")
        self.labels["ga"].config(image="")

        h, w = self.gray.shape
        self.metrics["original"].config(text=f"{w}×{h} px, {os.path.basename(path)}")
        self.btn_run.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self.status.config(text="Изображение загружено. Нажмите «Подобрать пороги».")

    # ------------------------------------------------------------------ #

    def start_search(self):
        """Запускает генетический алгоритм в отдельном потоке."""
        if self.gray is None:
            return
        try:
            cfg = self._read_config()
        except ValueError as err:
            messagebox.showwarning("Проверьте параметры", str(err))
            return

        self.btn_run.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.status.config(text="Поиск порогов…")

        gray = self.gray

        def worker():
            try:
                def progress(gen, total, best, score):
                    self._queue.put(("progress", gen, total, best, score))
                result = evolve(gray, cfg, progress=progress)
                self._queue.put(("done", result, cfg))
            except Exception as err:                      # noqa: BLE001
                self._queue.put(("error", err))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll)

    def _poll(self):
        """Забирает сообщения из рабочего потока и обновляет интерфейс."""
        busy = True
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "progress":
                    _, gen, total, best, score = msg
                    self.status.config(
                        text=f"Поколение {gen}/{total} · лучшие пороги "
                             f"({best[0]}, {best[1]}) · оценка {score:+.4f}")
                elif msg[0] == "done":
                    self._finish(msg[1], msg[2])
                    busy = False
                elif msg[0] == "error":
                    messagebox.showerror("Ошибка расчёта", str(msg[1]))
                    self.status.config(text="Расчёт прерван ошибкой")
                    busy = False
        except queue.Empty:
            pass

        if busy:
            self.root.after(50, self._poll)
        else:
            self.btn_run.config(state=tk.NORMAL)
            self.btn_open.config(state=tk.NORMAL)

    def _finish(self, result, cfg: GAConfig):
        """Показывает результат работы алгоритма."""
        self.edges_ga = E.canny(self.gray, result.t1, result.t2)
        self._set_preview("ga", self.edges_ga)
        self.captions["ga"].config(text=f"Кэнни + ГА ({result.t1}, {result.t2})")
        self.metrics["ga"].config(text=result.metrics.as_text())

        m_def = E.quality(self.edges_default, cfg.target_ratio)
        self.metrics["default"].config(text=m_def.as_text())
        self.btn_save.config(state=tk.NORMAL)
        self.status.config(
            text=f"Готово: пороги ({result.t1}, {result.t2}); оценка качества "
                 f"{result.metrics.score:+.4f} против {m_def.score:+.4f} без ГА; "
                 f"вычислений Кэнни — {result.evaluations}")

    def save_result(self):
        """Сохраняет обе карты контуров рядом с исходным файлом."""
        if self.edges_ga is None:
            return
        stem, _ = os.path.splitext(self.image_path)
        paths = []
        for suffix, array in (("_canny_default.png", self.edges_default),
                              ("_canny_ga.png", self.edges_ga)):
            p = stem + suffix
            Image.fromarray(array).save(p)
            paths.append(os.path.basename(p))
        self.status.config(text="Сохранено: " + ", ".join(paths))


def run_gui(cfg: GAConfig = GAConfig()):
    root = tk.Tk()
    App(root, cfg)
    root.mainloop()


if __name__ == "__main__":
    run_gui()

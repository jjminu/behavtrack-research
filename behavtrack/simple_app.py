"""A single-screen group approach monitor: live preview, food mark, start/stop."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from .ai_live import AIWorker
from .group_approach import Observation, build_summary, summary
from .sources import open_camera

ROOT = Path(__file__).resolve().parents[1]


def image_point(x, y, viewport):
    """Map a letterboxed preview point to the full image, rejecting the margins."""
    left, top, width, height = viewport
    if width <= 0 or height <= 0 or not left <= x <= left + width or not top <= y <= top + height:
        return None
    return ((x - left) / width, (y - top) / height)


class SimpleApp:
    def __init__(self, root):
        self.root = root
        root.title("BehavTrack · 그룹 사료 접근")
        root.geometry("1060x800")
        root.minsize(800, 680)
        root.configure(bg="#f2f6f4")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f2f6f4")
        style.configure("TLabel", background="#f2f6f4", foreground="#233b34", font=("Malgun Gothic", 10))
        style.configure("TButton", font=("Malgun Gothic", 10), padding=(12, 8))
        style.configure("Primary.TButton", background="#17765e", foreground="white", font=("Malgun Gothic", 12, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#bbc9c3"), ("active", "#105d49")])
        style.configure("Title.TLabel", font=("Malgun Gothic", 21, "bold"))
        style.configure("Metric.TLabel", font=("Malgun Gothic", 18, "bold"))
        self.cap = self.frame = self.worker = self.session = None
        self.food = self.drag = None
        self.viewport = (0, 0, 0, 0)
        self.pending = None
        self.preparing = False
        self.last_sample = None
        self.stamps = deque(maxlen=45)
        self.output = ROOT / "results"
        self.interval = 5.0
        self.latest_report = None
        self.camera = tk.StringVar(value="0")
        self.animals = tk.StringVar(value="5")
        self.group = tk.StringVar(value="그룹 1")
        self.status = tk.StringVar(value="① 카메라를 켜 주세요")
        self.detail = tk.StringVar(value="카메라 화면 전체를 관찰합니다. 케이지 테두리를 그릴 필요가 없습니다.")
        self.current = tk.StringVar(value="대기 중")
        self.ratio = tk.StringVar(value="—")
        self.unknown = tk.StringVar(value="—")
        self.age = tk.StringVar(value="관찰을 시작하면 결과가 표시됩니다")
        self.elapsed = tk.StringVar(value="")
        self.log_lines = []
        self.image = None

        page = ttk.Frame(root, padding=(24, 18))
        page.pack(fill="both", expand=True)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(3, weight=1)
        header = ttk.Frame(page)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="그룹 사료 접근", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="전체 결과 보기", command=self.show_results).pack(side="right")
        ttk.Label(page, text="카메라 켜기  →  사료 주변을 한 번 드래그  →  관찰 시작").grid(row=1, column=0, sticky="w", pady=(4, 14))
        controls = ttk.Frame(page)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(controls, text="카메라").pack(side="left")
        self.camera_field = ttk.Combobox(controls, textvariable=self.camera, values=[str(i) for i in range(6)], width=4, state="readonly")
        self.camera_field.pack(side="left", padx=(8, 6))
        self.open_button = ttk.Button(controls, text="① 카메라 켜기", command=self.open)
        self.open_button.pack(side="left", padx=(0, 20))
        ttk.Label(controls, text="그룹 이름").pack(side="left")
        self.group_field = ttk.Entry(controls, textvariable=self.group, width=17)
        self.group_field.pack(side="left", padx=(8, 16))
        ttk.Label(controls, text="전체 마릿수").pack(side="left")
        self.animals_field = ttk.Spinbox(controls, from_=1, to=20, textvariable=self.animals, width=4)
        self.animals_field.pack(side="left", padx=8)

        self.canvas = tk.Canvas(page, background="#192c26", highlightthickness=0, height=200, width=200)
        self.canvas.grid(row=3, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.begin_mark)
        self.canvas.bind("<B1-Motion>", self.move_mark)
        self.canvas.bind("<ButtonRelease-1>", self.end_mark)
        self.canvas.bind("<Configure>", lambda event: self.paint())
        self.canvas.create_text(400, 160, text="카메라를 켜면 여기에 실시간 영상이 보입니다", fill="#d5e7df", font=("Malgun Gothic", 13))
        ttk.Label(page, textvariable=self.detail, wraplength=720).grid(row=4, column=0, sticky="w", pady=(8, 8))
        metrics = ttk.Frame(page)
        metrics.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        for title, variable in [("최근 분석 · 사료 주변", self.current), ("접근 관측 비율", self.ratio), ("판정 불가 표본", self.unknown)]:
            card = ttk.Frame(metrics, padding=(0, 4, 30, 4))
            card.pack(side="left", fill="x", expand=True)
            ttk.Label(card, text=title).pack(anchor="w")
            ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w")
        ttk.Label(page, textvariable=self.age).grid(row=6, column=0, sticky="w")
        action = ttk.Frame(page)
        action.grid(row=7, column=0, sticky="ew", pady=(12, 8))
        self.start_button = ttk.Button(action, text="③ 관찰 시작", command=self.start, style="Primary.TButton", state="disabled")
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(action, text="종료하고 저장", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=10)
        self.clear_button = ttk.Button(action, text="사료 위치 다시 표시", command=self.clear_mark, state="disabled")
        self.clear_button.pack(side="left")
        status_row = ttk.Frame(page)
        status_row.grid(row=8, column=0, sticky="ew")
        ttk.Label(status_row, textvariable=self.status, font=("Malgun Gothic", 11, "bold")).pack(side="left")
        ttk.Label(status_row, textvariable=self.elapsed).pack(side="right")
        ttk.Label(page, text="접근 = 표시한 사료 주변 안에 몸 중심이 있음 · 관측 비율은 판정 가능한 표본 기준이며 섭식량·체류 시간이 아닙니다.",
                  wraplength=720).grid(row=9, column=0, sticky="w", pady=(8, 0))
        footer = ttk.Frame(page)
        footer.grid(row=10, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(footer, text="현재 모델: 위에서 촬영한 마우스용 · 학술/비상업용").pack(side="left")
        ttk.Button(footer, text="저장 위치·설정", command=self.settings).pack(side="right")
        ttk.Button(footer, text="이전 기능", command=self.advanced).pack(side="right", padx=6)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(40, self.tick)

    @property
    def busy(self):
        return self.preparing or self.session is not None

    def log(self, text):
        self.log_lines.append(text)
        print(text, flush=True)

    def set_buttons(self):
        enabled = "disabled" if self.busy else "normal"
        for widget in (self.open_button, self.group_field, self.animals_field):
            widget.configure(state=enabled)
        self.camera_field.configure(state="disabled" if self.busy else "readonly")
        self.start_button.configure(state="normal" if not self.busy and self.food and self.frame is not None else "disabled")
        self.clear_button.configure(state="normal" if not self.busy and self.food else "disabled")
        self.stop_button.configure(state="normal" if self.busy else "disabled",
                                   text="준비 취소" if self.preparing else "종료하고 저장")

    def open(self):
        if self.busy:
            return
        self.disconnect()
        self.status.set("카메라 연결 중…")
        self.root.update_idletasks()
        try:
            self.cap = open_camera(int(self.camera.get()))
            ok, self.frame = self.cap.read()
            if not ok:
                raise ValueError("카메라에서 영상을 읽을 수 없습니다.")
            self.status.set("② 영상에서 사료 주변을 드래그해 주세요")
            self.detail.set("사료와 마우스가 다가서는 주변 공간을 함께 표시하세요. 화면 전체에서 마우스를 찾습니다.")
            self.open_button.configure(text="카메라 다시 연결")
        except (Exception, SystemExit) as exc:
            self.disconnect()
            self.status.set("카메라 연결을 확인해 주세요")
            messagebox.showerror("카메라 연결", "다른 카메라 번호를 선택해 다시 켜 주세요.\n카메라를 쓰는 다른 앱도 닫아 주세요.\n\n" + str(exc), parent=self.root)
        self.set_buttons()

    def disconnect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = self.frame = self.food = self.drag = None
        self.stamps.clear()
        self.canvas.delete("all")

    def begin_mark(self, event):
        if not self.busy and self.frame is not None:
            point = image_point(event.x, event.y, self.viewport)
            self.drag = [point, point] if point else None

    def move_mark(self, event):
        if self.drag and not self.busy:
            left, top, width, height = self.viewport
            self.drag[1] = (max(0, min(1, (event.x-left)/width)), max(0, min(1, (event.y-top)/height)))
            self.paint()

    def end_mark(self, event):
        if not self.drag or self.busy:
            return
        self.move_mark(event)
        a, b = self.drag
        self.drag = None
        rect = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        if rect[2]-rect[0] < .01 or rect[3]-rect[1] < .01:
            self.detail.set("클릭한 채로 끌어서 사료 주변에 사각형을 만들어 주세요.")
            return
        self.food = rect
        self.status.set("③ 준비됐습니다. 관찰 시작을 눌러 주세요")
        self.detail.set("초록색 영역이 사료 주변입니다. 케이지 전체는 자동으로 관찰합니다.")
        self.paint()
        self.set_buttons()

    def clear_mark(self):
        if self.busy:
            return
        self.food = self.drag = None
        self.status.set("② 사료 주변을 다시 드래그해 주세요")
        self.set_buttons()
        self.paint()

    def paint(self):
        if self.frame is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = self.frame.shape[:2]
        scale = min(cw/w, ch/h)
        width, height = max(1, int(w*scale)), max(1, int(h*scale))
        left, top = (cw-width)/2, (ch-height)/2
        self.viewport = (left, top, width, height)
        image = cv2.resize(self.frame, (width, height), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", image)
        if not ok:
            return
        self.image = tk.PhotoImage(data=png.tobytes(), master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.image, anchor="nw")
        rect = self.food
        if self.drag:
            a, b = self.drag
            rect = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        if rect:
            x1, y1, x2, y2 = rect
            self.canvas.create_rectangle(left+x1*width, top+y1*height, left+x2*width, top+y2*height, outline="#30ee9b", width=3)
            self.canvas.create_text(left+x1*width+5, top+y1*height+5, anchor="nw", text="사료 주변", fill="#30ee9b", font=("Malgun Gothic", 11, "bold"))

    def start(self):
        if self.busy or self.frame is None or not self.food:
            return
        try:
            animals = int(self.animals.get())
            if not 1 <= animals <= 20:
                raise ValueError
        except ValueError:
            messagebox.showerror("마릿수", "케이지의 전체 마릿수를 1~20 사이로 입력해 주세요.", parent=self.root)
            return
        executable = ROOT / ".ai_venv" / "Scripts" / "python.exe"
        if not executable.is_file():
            messagebox.showerror("분석 준비", "프로그램을 닫고 실행하기.cmd로 다시 열어 주세요. AI 환경이 설치됩니다.", parent=self.root)
            return
        try:
            self.worker = AIWorker(executable, ["--detect-only", "--animals", str(animals)])
        except OSError as exc:
            messagebox.showerror("분석 준비", str(exc), parent=self.root)
            return
        self.preparing = True
        self.expected = animals
        self.group_name = self.group.get().strip() or "이름 없는 그룹"
        self.status.set("분석 준비 중… 준비되면 자동으로 녹화를 시작합니다")
        self.current.set("준비 중")
        self.ratio.set("—")
        self.unknown.set("—")
        self.last_sample = None
        self.age.set("마우스 몸 위치를 찾는 중입니다. 관절은 분석하지 않습니다.")
        self.set_buttons()

    def ready(self):
        fps = ((len(self.stamps)-1)/(self.stamps[-1]-self.stamps[0])) if len(self.stamps) > 2 and self.stamps[-1] > self.stamps[0] else 20
        self.session = Observation(self.output, self.group_name, self.expected, self.food, self.frame.shape[1::-1],
                                   fps, self.interval, int(self.camera.get()))
        self.started = time.monotonic()
        self.last_request = -self.interval
        self.pending = None
        self.preparing = False
        self.status.set("● 관찰 중 · 원본 영상과 접근 결과를 저장합니다")
        self.log(f"저장 위치: {self.session.folder}")
        self.set_buttons()

    def update_metrics(self, result=None):
        s = summary(self.session.samples)
        self.ratio.set("—" if s["approach_pct"] is None else f'{s["approach_pct"]:.1f}%')
        self.unknown.set(f'{s["unknown_samples"]} / {s["samples"]}')
        if result:
            self.current.set("판정 불가" if result["count"] is None else
                             "접근 없음" if result["count"] == 0 else f'{result["count"]}마리 접근')

    def tick(self):
        try:
            if self.cap is not None:
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("카메라 연결이 끊어졌습니다.")
                if self.frame is not None and frame.shape != self.frame.shape:
                    raise RuntimeError("카메라 영상 크기가 바뀌었습니다. 다시 연결해 주세요.")
                self.frame = frame
                now = time.monotonic()
                self.stamps.append(now)
                if self.worker:
                    while True:
                        try:
                            kind, event = self.worker.events.get_nowait()
                        except queue.Empty:
                            break
                        if kind == "error":
                            raise RuntimeError("분석기를 시작하거나 실행하지 못했습니다: " + str(event))
                        if kind == "ready":
                            self.ready()
                        elif kind == "result" and self.session is not None:
                            fid, elapsed, prediction = event
                            if self.pending != (fid, elapsed):
                                raise RuntimeError("분석 결과의 촬영 시각이 일치하지 않습니다.")
                            result = self.session.accept(fid, elapsed, prediction)
                            self.pending = None
                            self.last_sample = self.started + elapsed
                            self.update_metrics(result)
                if self.session:
                    elapsed = max(0.0, now-self.started)
                    fid = self.session.record(frame, elapsed)
                    self.elapsed.set(f"녹화 {int(elapsed)//60:02d}:{int(elapsed)%60:02d}")
                    if self.pending is None and elapsed-self.last_request >= self.interval:
                        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not ok:
                            raise RuntimeError("분석할 영상을 만들지 못했습니다.")
                        self.worker.requests.put_nowait((fid, elapsed, encoded.tobytes()))
                        self.pending = (fid, elapsed)
                        self.last_request = elapsed
                    age = "첫 결과를 분석 중입니다" if self.last_sample is None else f"최근 결과: {max(0, now-self.last_sample):.1f}초 전 촬영 화면 기준"
                    self.age.set(age + (" · 다음 표본 분석 중" if self.pending else ""))
                self.paint()
        except (Exception, SystemExit) as exc:
            self.log(str(exc))
            self.stop(error=str(exc))
            self.disconnect()
            self.status.set("관찰이 중단되었습니다. 연결을 확인해 주세요")
            self.set_buttons()
            messagebox.showerror("관찰 중단", str(exc) + "\n촬영된 결과는 저장 폴더에서 확인할 수 있습니다.", parent=self.root)
        self.root.after(40, self.tick)

    def stop(self, error=None):
        if not self.busy and self.worker is None:
            return
        cancelled = self.preparing
        if self.worker:
            self.worker.close()
            self.worker = None
        self.preparing = False
        if self.session:
            if self.pending:
                self.session.accept(*self.pending, {}, reason="inference_incomplete")
            self.pending = None
            self.update_metrics()
            try:
                self.session.close(error)
                self.latest_report = self.session.folder / "report.html"
                build_summary(self.output)
                self.status.set("저장 완료 · 전체 결과 보기에서 확인하세요")
            finally:
                self.session = None
        elif cancelled:
            self.status.set("준비를 취소했습니다. 관찰 시작으로 다시 시작할 수 있습니다")
            self.current.set("대기 중")
        self.age.set("관찰 종료 · 위 수치는 마지막 관찰 결과입니다")
        self.set_buttons()

    def show_results(self):
        try:
            report = build_summary(self.output)
            os.startfile(str(report))
        except (OSError, ValueError) as exc:
            messagebox.showinfo("전체 결과", str(exc), parent=self.root)

    def settings(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("저장 위치 · 설정")
        dialog.geometry("620x360")
        panel = ttk.Frame(dialog, padding=20)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="결과 저장 위치").pack(anchor="w")
        ttk.Label(panel, text=str(self.output), wraplength=560).pack(anchor="w", pady=8)
        def choose():
            if self.busy:
                return
            selected = filedialog.askdirectory(parent=dialog, title="결과 저장 위치")
            if selected:
                self.output = Path(selected)
                dialog.destroy()
        ttk.Button(panel, text="저장 위치 변경", command=choose, state="disabled" if self.busy else "normal").pack(anchor="w")
        ttk.Label(panel, text="CPU 분석은 최소 5초 간격입니다. 처리 속도에 따라 더 길어질 수 있습니다.\n"
                  "영상은 계속 녹화하고, 분석 시각과 판정 불가를 함께 저장합니다.\n"
                  "겹침·가림·밝은 배경에서는 오인식할 수 있으므로 원본 영상과 비교하세요.", wraplength=560).pack(anchor="w", pady=14)
        ttk.Button(panel, text="저장 폴더 열기", command=self.open_output).pack(anchor="w")
        ttk.Label(panel, text="\n".join(self.log_lines[-2:]), wraplength=560).pack(anchor="w", pady=10)

    def open_output(self):
        self.output.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.output))

    def advanced(self):
        if self.busy:
            messagebox.showinfo("관찰 중", "관찰을 종료한 뒤 이전 기능을 열어 주세요.", parent=self.root)
            return
        self.disconnect()
        self.set_buttons()
        subprocess.Popen([sys.executable, str(ROOT / "windows_launcher.py"), "--advanced"], cwd=ROOT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def close(self):
        self.stop()
        self.disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    SimpleApp(root)
    root.mainloop()

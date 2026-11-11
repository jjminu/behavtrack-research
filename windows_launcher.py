"""Windows starter for the copied behavtrack project."""
from __future__ import annotations

import os
import math
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent


class Launcher:
    def __init__(self, root):
        self.root = root
        root.title("BehavTrack - 마우스 행동 관찰")
        root.geometry("900x760")
        root.minsize(760, 680)
        self.process = None
        self.messages = queue.Queue()
        self.values = {name: tk.StringVar() for name in
                       ("video", "output", "cage", "group", "animals", "interval", "recorded")}
        self.resume = tk.BooleanVar()
        self.live_values = {name: tk.StringVar(value=value) for name, value in {
            "camera": "0", "width": "", "height": "", "duration": "",
            "config": "", "output": str(ROOT / "results"), "color": "dark", "animal_id": "", "group": "",
        }.items()}
        self.group_values = {name: tk.StringVar(value=value) for name, value in {"animals": "", "cage_id": "", "ai_interval": "5"}.items()}

        top = ttk.Frame(root, padding=12)
        top.pack(fill="both", expand=True)
        ttk.Label(top, text="BehavTrack 윈도우 시작 화면", font=("Malgun Gothic", 15, "bold")).pack(anchor="w")
        ttk.Label(top, text="처음에는 가상 영상으로 실행을 확인하세요. 실제 연구에 사용하기 전 수동 검증이 필요합니다.", wraplength=790).pack(anchor="w", pady=(4, 10))
        actions = ttk.Frame(top)
        actions.pack(fill="x")
        ttk.Button(actions, text="가상 영상 체험", command=self.demo).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="카메라 확인", command=lambda: self.run(["cameras"])).pack(side="left", padx=6)
        ttk.Button(actions, text="윈도우 사용 설명서", command=self.open_guide).pack(side="left", padx=6)
        ttk.Button(actions, text="고급 명령 창", command=self.command_prompt).pack(side="left", padx=6)

        tabs = ttk.Notebook(top)
        tabs.pack(fill="x", pady=12)
        live = ttk.Frame(tabs, padding=10)
        tabs.add(live, text="웹캠 실시간 추적")
        group_live = ttk.Frame(tabs, padding=10)
        tabs.add(group_live, text="공동 사육 웹캠 (실험적)")
        form = ttk.Frame(tabs, padding=10)
        tabs.add(form, text="공동 사육 영상 수동 채점")

        ttk.Label(live, text="카메라 번호").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(live, textvariable=self.live_values["camera"], width=8).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(live, text="관찰 영역 크기 (cm, 가로 × 세로)").grid(row=1, column=0, sticky="w", pady=4)
        size_row = ttk.Frame(live)
        size_row.grid(row=1, column=1, sticky="w", padx=8)
        ttk.Entry(size_row, textvariable=self.live_values["width"], width=9).pack(side="left")
        ttk.Label(size_row, text=" × ").pack(side="left")
        ttk.Entry(size_row, textvariable=self.live_values["height"], width=9).pack(side="left")
        ttk.Label(live, text="동물 색 (바닥 대비)").grid(row=2, column=0, sticky="w", pady=4)
        colors = ttk.Combobox(live, state="readonly", values=("어두움", "밝음", "자동 (명암 차이)"), width=20)
        colors.current(0)
        colors.grid(row=2, column=1, sticky="w", padx=8)
        colors.bind("<<ComboboxSelected>>", lambda event: self.live_values["color"].set(("dark", "light", "any")[colors.current()]))
        ttk.Label(live, text="관찰·사료 구역 설정 파일").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(live, textvariable=self.live_values["config"]).grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Button(live, text="기존 설정 찾기", command=self.choose_config).grid(row=3, column=2)
        ttk.Button(live, text="1. 관찰·사료 구역 설정", command=self.setup_live).grid(row=4, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(live, text="녹화 시간 (초, 비우면 수동 종료)").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(live, textvariable=self.live_values["duration"], width=12).grid(row=5, column=1, sticky="w", padx=8)
        ttk.Label(live, text="결과 저장 폴더").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(live, textvariable=self.live_values["output"]).grid(row=6, column=1, sticky="ew", padx=8)
        ttk.Button(live, text="찾아보기", command=self.choose_live_output).grid(row=6, column=2)
        ttk.Button(live, text="2. 웹캠 실시간 추적 시작", command=self.start_live).grid(row=7, column=1, sticky="w", padx=8, pady=6)
        ttk.Button(live, text="누적 분석 보기", command=self.show_camera_summary).grid(row=7, column=2, sticky="w", pady=6)
        ttk.Label(live, text="동물 ID / 군 코드 (선택)").grid(row=8, column=0, sticky="w", pady=4)
        ids = ttk.Frame(live)
        ids.grid(row=8, column=1, sticky="w", padx=8)
        ttk.Entry(ids, textvariable=self.live_values["animal_id"], width=20).pack(side="left")
        ttk.Label(ids, text=" / ").pack(side="left")
        ttk.Entry(ids, textvariable=self.live_values["group"], width=20).pack(side="left")
        ttk.Label(live, text="한 관찰 영역에 한 마리만 추적합니다. 사료 구역 체류는 실제 섭식 판정이 아닙니다.", wraplength=780).grid(row=9, column=0, columnspan=3, sticky="w")
        live.columnconfigure(1, weight=1)
        ttk.Label(group_live, text="위에서 촬영한 영상으로 사료 구역 접근 마릿수를 추정합니다. 실제 섭식이나 개체별 추적은 아닙니다.", wraplength=780).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(group_live, text="카메라 번호").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(group_live, textvariable=self.live_values["camera"], width=10).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Label(group_live, text="관찰·사료 구역 설정 파일").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(group_live, textvariable=self.live_values["config"]).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(group_live, text="기존 설정 찾기", command=self.choose_config).grid(row=2, column=2)
        ttk.Label(group_live, text="처음에는 웹캠 실시간 추적 탭에서 관찰·사료 구역 설정을 만드세요.").grid(row=3, column=1, sticky="w", padx=8)
        ttk.Label(group_live, text="케이지의 마우스 수 (2마리 이상)").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(group_live, textvariable=self.group_values["animals"], width=10).grid(row=4, column=1, sticky="w", padx=8)
        ttk.Label(group_live, text="케이지 ID / 군 코드 (선택)").grid(row=5, column=0, sticky="w", pady=4)
        group_ids = ttk.Frame(group_live)
        group_ids.grid(row=5, column=1, sticky="w", padx=8)
        ttk.Entry(group_ids, textvariable=self.group_values["cage_id"], width=20).pack(side="left")
        ttk.Label(group_ids, text=" / ").pack(side="left")
        ttk.Entry(group_ids, textvariable=self.live_values["group"], width=20).pack(side="left")
        ttk.Label(group_live, text="녹화 시간 (초, 비우면 수동 종료)").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(group_live, textvariable=self.live_values["duration"], width=12).grid(row=6, column=1, sticky="w", padx=8)
        ttk.Label(group_live, text="결과 저장 폴더").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(group_live, textvariable=self.live_values["output"]).grid(row=7, column=1, sticky="ew", padx=8)
        ttk.Button(group_live, text="찾아보기", command=self.choose_live_output).grid(row=7, column=2)
        ttk.Button(group_live, text="공동 사육 실시간 추정 시작", command=self.start_group_live).grid(row=8, column=1, sticky="w", padx=8, pady=6)
        ttk.Button(group_live, text="공동 사육 누적 분석 보기", command=self.show_group_summary).grid(row=8, column=2, sticky="w", pady=6)
        ttk.Label(group_live, text="겹침·가림은 판정 불가로 표시될 수 있으며, 마릿수가 맞아도 오인식할 수 있습니다. 실험 환경에서 수동 검증하세요.", wraplength=780).grid(row=9, column=0, columnspan=3, sticky="w")
        ttk.Label(group_live, text="AI 표본 간격 (초)").grid(row=10, column=0, sticky="w", pady=4)
        ttk.Entry(group_live, textvariable=self.group_values["ai_interval"], width=10).grid(row=10, column=1, sticky="w", padx=8)
        ttk.Button(group_live, text="AI로 5마리 접근 추정 시작 (CPU)", command=self.start_group_ai).grid(row=11, column=1, sticky="w", padx=8, pady=4)
        ttk.Button(group_live, text="AI 누적 분석 보기", command=self.show_ai_group_summary).grid(row=11, column=2, sticky="w", pady=4)
        ttk.Label(group_live, text="AI는 코 위치를 표본별로 추정합니다. CPU 속도에 따라 간격이 길어집니다. 모델은 학술·비상업적 사용에 한정됩니다.", wraplength=780).grid(row=12, column=0, columnspan=3, sticky="w")
        group_live.columnconfigure(1, weight=1)
        fields = [
            ("영상 파일", "video", self.choose_video),
            ("결과 폴더 (새 폴더)", "output", self.choose_output),
            ("케이지 ID", "cage", None),
            ("군 코드", "group", None),
            ("마우스 수 (1~9)", "animals", None),
            ("표본 간격 (초)", "interval", None),
            ("촬영 시작 일시 (선택)", "recorded", None),
        ]
        for row, (label, key, browse) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(form, textvariable=self.values[key]).grid(row=row, column=1, sticky="ew", padx=8, pady=3)
            if browse:
                ttk.Button(form, text="찾아보기", command=browse).grid(row=row, column=2)
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="촬영 일시 예: 2026-10-28T20:00:00+09:00").grid(row=7, column=1, sticky="w", padx=8)
        ttk.Checkbutton(form, text="이전에 채점한 결과 이어서 하기 (같은 조건)", variable=self.resume).grid(row=8, column=1, sticky="w", padx=8, pady=4)
        ttk.Button(form, text="수동 채점 시작", command=self.score).grid(row=9, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(form, text="접근과 섭식을 각각 채점합니다. 판정이 어려우면 u를 누르세요.").grid(row=10, column=1, sticky="w", padx=8)

        ttk.Label(top, text="실행 상태와 결과").pack(anchor="w")
        log_frame = ttk.Frame(top)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self, args, on_success=None):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "behavtrack", *args], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            messagebox.showerror("실행 오류", str(exc))
            return
        self.append("\n작업을 시작했습니다.\n")
        process = self.process

        def read_output():
            for line in process.stdout:
                self.messages.put(("output", line))
            self.messages.put(("done", (process.wait(), on_success)))

        threading.Thread(target=read_output, daemon=True).start()

    def poll(self):
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "output":
                self.append(value)
            else:
                self.process = None
                code, on_success = value
                self.append("\n작업 완료.\n" if code == 0 else f"\n작업이 종료되었습니다 (오류 코드 {code}). 위 내용을 확인하세요.\n")
                if code == 0 and on_success:
                    on_success()
        self.root.after(100, self.poll)

    def camera_number(self):
        try:
            camera = int(self.live_values["camera"].get().strip())
            if camera < 0:
                raise ValueError
            return camera
        except ValueError:
            messagebox.showerror("입력 확인", "카메라 번호에 0 이상의 정수를 입력하세요.")
            return None

    def choose_config(self):
        path = filedialog.askopenfilename(title="관찰 영역 설정 파일 선택", filetypes=[("설정 파일", "*.json")])
        if path:
            self.live_values["config"].set(path)

    def choose_live_output(self):
        path = filedialog.askdirectory(title="웹캠 결과를 저장할 폴더 선택")
        if path:
            self.live_values["output"].set(path)

    def setup_live(self):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        camera = self.camera_number()
        if camera is None:
            return
        try:
            width = float(self.live_values["width"].get())
            height = float(self.live_values["height"].get())
            if not all(math.isfinite(x) and x > 0 for x in (width, height)):
                raise ValueError
        except ValueError:
            messagebox.showerror("입력 확인", "관찰 영역의 실제 가로와 세로를 양수 cm로 입력하세요.")
            return
        path = filedialog.asksaveasfilename(title="관찰 영역 설정 저장", defaultextension=".json",
                                            initialdir=str(ROOT), initialfile="camera_setup.json",
                                            filetypes=[("설정 파일", "*.json")])
        if not path:
            return
        messagebox.showinfo("영역 설정 방법", "영상 창에서 관찰 영역의 네 꼭짓점을 클릭하세요.\n\n"
                            "이어서 사료 접근 구역의 경계를 순서대로 3점 이상 클릭하고 Enter를 누르세요.\n\n"
                            "확인 화면에서 Enter를 누르면 저장합니다. 오른쪽 클릭은 마지막 점 취소, q는 중단입니다.")
        self.append("관찰 영역의 4개 꼭짓점 → 사료 구역의 경계점(3개 이상) → Enter → 확인 후 Enter 순서로 설정하세요.\n")
        args = ["setup", "-p", "open_field", "--camera", str(camera), "--arena-size", f"{width}x{height}",
                "--color", self.live_values["color"].get(), "--food-zone", "-o", path]
        self.run(args, on_success=lambda: self.live_values["config"].set(path))

    def start_live(self):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        camera = self.camera_number()
        if camera is None:
            return
        config = self.live_values["config"].get().strip()
        output = self.live_values["output"].get().strip()
        if not config or not Path(config).is_file():
            messagebox.showerror("입력 확인", "먼저 관찰·사료 구역을 설정하거나 기존 설정 파일을 선택하세요.")
            return
        if not output:
            messagebox.showerror("입력 확인", "결과 저장 폴더를 입력하세요.")
            return
        duration = self.live_values["duration"].get().strip()
        if duration:
            try:
                seconds = float(duration)
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("입력 확인", "녹화 시간은 양수로 입력하거나 비워 두세요.")
                return
        messagebox.showinfo("실시간 추적 방법", "영상 창을 클릭한 뒤 빈 관찰 영역에서 b를 눌러 배경을 촬영하세요.\n\n"
                            "배경 저장이 끝나면 동물을 넣고 s를 눌러 녹화를 시작하세요.\n"
                            "q를 누르면 저장 후 종료하며 분석 보고서가 생성됩니다.")
        self.append("영상 창에서 빈 영역의 배경을 b로 촬영하고, 동물을 넣은 뒤 s로 녹화를 시작하세요. q는 종료입니다.\n")
        args = ["track", "-c", config, "--camera", str(camera), "-o", output, "--annotated"]
        for field, option in (("animal_id", "--animal-id"), ("group", "--group")):
            value = self.live_values[field].get().strip()
            if value:
                args += [option, value]
        if duration:
            args += ["--duration", duration]
        output_path = Path(output)
        before = set(output_path.glob("*/session.json")) if output_path.is_dir() else set()
        def update_if_recorded():
            after = set(output_path.glob("*/session.json")) if output_path.is_dir() else set()
            if after - before:
                self.append("새 촬영 결과를 누적 보고서에 반영합니다.\n")
                self.run(["camera-summary", output])
        self.run(args, on_success=update_if_recorded)

    def show_camera_summary(self):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        folder = self.live_values["output"].get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showerror("입력 확인", "웹캠 결과가 저장된 폴더를 선택하세요.")
            return
        report = Path(folder).resolve() / "누적_분석" / "누적_보고서.html"
        def open_report():
            try:
                os.startfile(str(report))
            except OSError as exc:
                messagebox.showerror("보고서 열기 오류", str(exc))
        self.run(["camera-summary", folder], on_success=open_report)

    def start_group_live(self):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        camera = self.camera_number()
        if camera is None:
            return
        try:
            animals = int(self.group_values["animals"].get().strip())
            if animals < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("입력 확인", "케이지의 마우스 수를 2 이상의 정수로 입력하세요.")
            return
        config = self.live_values["config"].get().strip()
        output = self.live_values["output"].get().strip()
        if not config or not Path(config).is_file():
            messagebox.showerror("입력 확인", "먼저 사료 구역이 포함된 설정 파일을 선택하세요.")
            return
        if not output:
            messagebox.showerror("입력 확인", "결과 저장 폴더를 입력하세요.")
            return
        duration = self.live_values["duration"].get().strip()
        if duration:
            try:
                seconds = float(duration)
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("입력 확인", "녹화 시간은 양수로 입력하거나 비워 두세요.")
                return
        messagebox.showinfo("공동 사육 추정 방법", "위에서 고정 촬영하세요. 영상 창의 빈 케이지에서 b를 눌러 배경을 촬영합니다.\n\n"
                            "배경 저장이 끝나면 동물을 넣고 s를 눌러 녹화합니다. q는 저장 후 종료합니다.\n\n"
                            "접근 마릿수는 실험적 추정이며, 실제 섭식이 아닙니다. 겹침·가림과 오인식을 수동으로 확인하세요.")
        self.append("공동 사육: 빈 케이지에서 b → 동물을 넣은 뒤 s → q로 종료. UNKNOWN은 판정 불가입니다.\n")
        args = ["group-live", "-c", config, "--camera", str(camera), "--animals", str(animals), "-o", output,
                "--cage-id", self.group_values["cage_id"].get().strip(), "--group", self.live_values["group"].get().strip()]
        if duration:
            args += ["--duration", duration]
        output_path = Path(output)
        before = set(output_path.glob("*/group_session.json")) if output_path.is_dir() else set()
        def update_if_recorded():
            after = set(output_path.glob("*/group_session.json")) if output_path.is_dir() else set()
            if after - before:
                self.append("새 공동 사육 결과를 누적 보고서에 반영합니다.\n")
                self.run(["group-summary", output])
        self.run(args, on_success=update_if_recorded)

    def show_group_summary(self):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        folder = self.live_values["output"].get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showerror("입력 확인", "공동 사육 결과가 저장된 폴더를 선택하세요.")
            return
        report = Path(folder).resolve() / "공동사육_누적_분석" / "공동사육_누적_보고서.html"
        def open_report():
            try:
                os.startfile(str(report))
            except OSError as exc:
                messagebox.showerror("보고서 열기 오류", str(exc))
        self.run(["group-summary", folder], on_success=open_report)

    def start_group_ai(self):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        camera = self.camera_number()
        if camera is None:
            return
        try:
            animals = int(self.group_values["animals"].get().strip())
            interval = float(self.group_values["ai_interval"].get().strip())
            if animals != 5 or not math.isfinite(interval) or interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("입력 확인", "AI 기능의 마우스 수는 5마리이며 표본 간격은 양수로 입력하세요.")
            return
        config = self.live_values["config"].get().strip()
        output = self.live_values["output"].get().strip()
        duration = self.live_values["duration"].get().strip()
        if not config or not Path(config).is_file() or not output:
            messagebox.showerror("입력 확인", "사료 구역 설정 파일과 결과 폴더를 확인하세요.")
            return
        if duration:
            try:
                if not math.isfinite(float(duration)) or float(duration) <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("입력 확인", "녹화 시간은 양수로 입력하거나 비워 두세요.")
                return
        if not (ROOT / ".ai_venv" / "Scripts" / "python.exe").is_file():
            messagebox.showerror("AI 환경 없음", "AI 모델 실행 환경이 없습니다. AI 기능이 포함된 설치 묶음을 사용하세요.")
            return
        messagebox.showinfo("AI 표본 분석", "AI 준비 완료 후 영상 창에서 s를 눌러 녹화하고 q로 종료합니다.\n\n"
            "코가 사료 구역 안에 있는 마릿수를 표본별로 추정합니다. CPU 속도에 따라 간격이 길어질 수 있습니다.\n"
            "겹침·가림을 놓칠 수 있으므로 실제 영상의 수동 판정과 비교하세요.")
        self.append("AI 모델 준비 중. 빈 케이지 배경은 필요하지 않습니다. 준비되면 영상 창에서 s, 종료는 q입니다.\n")
        args = ["group-ai-live", "-c", config, "--camera", str(camera), "-o", output,
                "--interval", str(interval), "--cage-id", self.group_values["cage_id"].get().strip(),
                "--group", self.live_values["group"].get().strip()]
        if duration:
            args += ["--duration", duration]
        output_path = Path(output)
        before = set(output_path.glob("*/ai_group_session.json")) if output_path.is_dir() else set()
        def update_if_recorded():
            after = set(output_path.glob("*/ai_group_session.json")) if output_path.is_dir() else set()
            if after - before:
                self.append("새 AI 표본 결과를 누적 보고서에 반영합니다.\n")
                self.run(["group-ai-summary", output])
        self.run(args, on_success=update_if_recorded)

    def show_ai_group_summary(self):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        folder = self.live_values["output"].get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showerror("입력 확인", "AI 결과가 저장된 폴더를 선택하세요.")
            return
        report = Path(folder).resolve() / "AI_공동사육_누적분석" / "AI_누적_보고서.html"
        def open_report():
            try:
                os.startfile(str(report))
            except OSError as exc:
                messagebox.showerror("보고서 열기 오류", str(exc))
        self.run(["group-ai-summary", folder], on_success=open_report)

    def demo(self):
        if self.process is not None:
            messagebox.showinfo("실행 중", "현재 작업이 끝난 다음에 다시 실행하세요.")
            return
        parent = filedialog.askdirectory(title="가상 영상 결과를 저장할 위치 선택")
        if not parent:
            return
        # Reserve a new directory so existing demo files cannot be overwritten.
        output = Path(parent) / ("behavtrack_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        try:
            output.mkdir()
        except OSError as exc:
            messagebox.showerror("폴더 오류", str(exc))
            return
        self.run(["demo", "-o", str(output)])
        self.append(f"가상 영상 결과: {output}\n")

    def choose_video(self):
        path = filedialog.askopenfilename(title="분석할 영상 선택", filetypes=[("영상 파일", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")])
        if path:
            self.values["video"].set(path)

    def choose_output(self):
        parent = filedialog.askdirectory(title="결과를 저장할 상위 폴더 선택")
        if parent:
            self.values["output"].set(str(Path(parent) / "feeding_result"))

    def score(self):
        v = {name: var.get().strip() for name, var in self.values.items()}
        if not all(v[name] for name in ("video", "output", "cage", "group", "animals", "interval")):
            messagebox.showerror("입력 확인", "영상, 결과 폴더, 케이지 ID, 군 코드, 마우스 수, 표본 간격을 모두 입력하세요.")
            return
        if not Path(v["video"]).is_file():
            messagebox.showerror("입력 확인", "영상 파일을 찾을 수 없습니다.")
            return
        output = Path(v["output"])
        if not self.resume.get() and output.exists():
            messagebox.showerror("입력 확인", "결과 폴더가 이미 있습니다. 새 폴더 이름을 입력하거나 이어하기를 선택하세요.")
            return
        args = ["feeding", "score", v["video"], "--output", v["output"], "--cage-id", v["cage"],
                "--group", v["group"], "--animals", v["animals"], "--interval", v["interval"]]
        if v["recorded"]:
            args += ["--recorded-at", v["recorded"]]
        if self.resume.get():
            args.append("--resume")
        self.run(args)

    def open_guide(self):
        subprocess.Popen(["notepad.exe", str(ROOT / "윈도우_사용설명서.md")])

    def command_prompt(self):
        subprocess.Popen(["cmd.exe", "/k", "echo Run: .venv\\Scripts\\python.exe -m behavtrack --help"], cwd=ROOT)

    def close(self):
        if self.process is not None:
            messagebox.showinfo("작업 실행 중", "작업을 먼저 종료하세요. 채점 창에서는 q를 누르면 완료한 표본이 저장됩니다.")
            return
        self.root.destroy()


if __name__ == "__main__":
    if "--advanced" in sys.argv:
        Launcher(tk.Tk())
        tk.mainloop()
    elif "--desktop" in sys.argv:
        from behavtrack.simple_app import main
        main()
    else:
        from behavtrack.research_web import main
        main()

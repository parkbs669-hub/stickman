"""Small local desktop launcher; rendering runs in a separate Python process."""
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent


def build_command(source, name, output, speed, height, stroke, left, two_handed, compare, strobe):
    command = [sys.executable, "-u", str(ROOT / "tennis_stickman_v12_4_upgraded.py"),
               source, name, "--output-dir", output, "--speed", speed,
               "--height", height, "--stroke", stroke, "--ensemble"]
    for flag, enabled in [("--left", left), ("--two-handed", two_handed),
                          ("--compare", compare), ("--strobe", strobe)]:
        if enabled:
            command.append(flag)
    return command


class Launcher:
    def __init__(self, window):
        self.window = window
        self.running = False
        self.events = queue.Queue()
        window.title("Tennis Stickman · v12.4 업그레이드")
        window.geometry("820x630")
        window.minsize(700, 520)
        self.source = tk.StringVar()
        self.name = tk.StringVar()
        self.output = tk.StringVar(value=str(ROOT / "renders"))
        self.speed = tk.StringVar(value="1")
        self.height = tk.StringVar(value="720")
        self.stroke = tk.StringVar(value="forehand")
        self.left = tk.BooleanVar()
        self.two = tk.BooleanVar()
        self.compare = tk.BooleanVar(value=True)
        self.strobe = tk.BooleanVar()
        frame = ttk.Frame(window, padding=20)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="테니스 영상을 스틱맨으로", font=("맑은 고딕", 18, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 15))
        self.controls = []
        for row, (label, variable) in enumerate([("영상 파일", self.source), ("결과 이름", self.name), ("저장 폴더", self.output)], 1):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
            field = ttk.Entry(frame, textvariable=variable)
            field.grid(row=row, column=1, sticky="ew", pady=5)
            self.controls.append(field)
        for row, action in [(1, self.pick_source), (3, self.pick_output)]:
            button = ttk.Button(frame, text="찾아보기", command=action)
            button.grid(row=row, column=2, padx=(8, 0))
            self.controls.append(button)
        options = ttk.Frame(frame)
        options.grid(row=4, column=0, columnspan=3, sticky="ew", pady=12)
        self.combos = []
        for col, (label, variable, values) in enumerate([
            ("배속", self.speed, ["0.25", "0.5", "1", "2"]),
            ("높이", self.height, ["720", "1080"]),
            ("동작", self.stroke, ["forehand", "backhand", "serve", "auto"]) ]):
            ttk.Label(options, text=label).grid(row=0, column=col*2, padx=(0, 7))
            box = ttk.Combobox(options, textvariable=variable, values=values, state="readonly", width=11)
            box.grid(row=0, column=col*2+1, padx=(0, 20))
            self.combos.append(box)
        toggles = ttk.Frame(frame)
        toggles.grid(row=5, column=0, columnspan=3, sticky="w")
        for text, variable in [("왼손잡이", self.left), ("양손 백핸드", self.two), ("원본 비교", self.compare), ("잔상", self.strobe)]:
            check = ttk.Checkbutton(toggles, text=text, variable=variable)
            check.pack(side="left", padx=(0, 18))
            self.controls.append(check)
        ttk.Label(frame, text="forehand: 포핸드 · backhand: 백핸드 · serve: 서브\n자동 임팩트는 추정치입니다. 같은 이름의 결과는 덮어쓰지 않습니다.").grid(row=6, column=0, columnspan=3, sticky="w", pady=12)
        actions = ttk.Frame(frame)
        actions.grid(row=7, column=0, columnspan=3, sticky="ew")
        self.start_button = ttk.Button(actions, text="영상 만들기", command=self.start)
        self.start_button.pack(side="left")
        ttk.Button(actions, text="결과 폴더 열기", command=self.open_output).pack(side="left", padx=8)
        self.status = ttk.Label(actions, text="준비")
        self.status.pack(side="right")
        self.log = tk.Text(frame, height=14, wrap="word", state="disabled", font=("맑은 고딕", 9))
        self.log.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        frame.rowconfigure(8, weight=1)
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.after(100, self.drain)

    def pick_source(self):
        path = filedialog.askopenfilename(filetypes=[("동영상", "*.mp4 *.mov *.mkv *.avi *.webm"), ("모든 파일", "*.*")])
        if path:
            self.source.set(path)
            self.name.set(Path(path).stem + "_upgraded")

    def pick_output(self):
        path = filedialog.askdirectory(initialdir=self.output.get())
        if path:
            self.output.set(path)

    def append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.running:
            return
        if not Path(self.source.get()).is_file() or not self.name.get().strip():
            messagebox.showerror("입력 확인", "영상 파일과 결과 이름을 지정하세요.")
            return
        command = build_command(self.source.get(), self.name.get(), self.output.get(),
                                self.speed.get(), self.height.get(), self.stroke.get(),
                                self.left.get(), self.two.get(), self.compare.get(), self.strobe.get())
        self.running = True
        for control in self.controls + self.combos + [self.start_button]:
            control.configure(state="disabled")
        self.status.configure(text="처리 중…")
        self.append("\n영상 생성을 시작합니다.\n")
        threading.Thread(target=self.worker, args=(command,), daemon=True).start()

    def worker(self, command):
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", creationflags=flags,
                                  cwd=ROOT) as process:
                for line in process.stdout:
                    self.events.put(("log", line))
                code = process.wait()
            self.events.put(("done", code))
        except Exception as exc:
            self.events.put(("log", f"실행 오류: {exc}\n"))
            self.events.put(("done", 1))

    def drain(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.append(value)
                else:
                    self.running = False
                    for control in self.controls + [self.start_button]:
                        control.configure(state="normal")
                    for control in self.combos:
                        control.configure(state="readonly")
                    self.status.configure(text="완료" if value == 0 else "오류 — 로그 확인")
        except queue.Empty:
            pass
        self.window.after(100, self.drain)

    def open_output(self):
        path = Path(self.output.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)

    def close(self):
        if self.running:
            messagebox.showinfo("영상 처리 중", "파일 저장이 끝나면 창을 닫을 수 있습니다.")
            return
        self.window.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    Launcher(window)
    window.mainloop()

import windows_launcher


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def launcher(tmp_path):
    ui = windows_launcher.Launcher.__new__(windows_launcher.Launcher)
    ui.process = None
    ui.live_values = {key: Value(value) for key, value in {
        "camera": "0", "width": "40", "height": "30", "duration": "60",
        "config": "", "output": str(tmp_path), "color": "dark", "animal_id": "M1", "group": "control",
    }.items()}
    ui.group_values = {"animals": Value("3"), "cage_id": Value("C1"), "ai_interval": Value("5")}
    ui.append = lambda text: None
    return ui


def test_setup_uses_drawn_food_zone_and_only_selects_after_success(tmp_path, monkeypatch):
    ui = launcher(tmp_path)
    path = str(tmp_path / "layout.json")
    calls = []
    ui.run = lambda args, on_success=None: calls.append((args, on_success))
    monkeypatch.setattr(windows_launcher.filedialog, "asksaveasfilename", lambda **kwargs: path)
    monkeypatch.setattr(windows_launcher.messagebox, "showinfo", lambda *args: None)
    ui.setup_live()
    args, callback = calls[0]
    assert "--food-zone" in args and args[args.index("--arena-size") + 1] == "40.0x30.0"
    assert ui.live_values["config"].get() == ""
    callback()
    assert ui.live_values["config"].get() == path


def test_start_live_uses_selected_configuration(tmp_path, monkeypatch):
    ui = launcher(tmp_path)
    config = tmp_path / "layout.json"
    config.write_text("{}", encoding="utf-8")
    ui.live_values["config"].set(str(config))
    calls = []
    ui.run = lambda args, on_success=None: calls.append((args, on_success))
    monkeypatch.setattr(windows_launcher.messagebox, "showinfo", lambda *args: None)
    ui.start_live()
    args, callback = calls[0]
    assert args[:3] == ["track", "-c", str(config)]
    assert args[args.index("--duration") + 1] == "60"
    assert "--annotated" in args
    assert args[args.index("--animal-id") + 1] == "M1"
    assert args[args.index("--group") + 1] == "control"
    session = tmp_path / "live_1"
    session.mkdir()
    (session / "session.json").write_text("{}", encoding="utf-8")
    callback()
    assert calls[1][0] == ["camera-summary", str(tmp_path)]


def test_start_group_live_selects_group_mode_and_updates_summary(tmp_path, monkeypatch):
    ui = launcher(tmp_path)
    config = tmp_path / "layout.json"
    config.write_text("{}", encoding="utf-8")
    ui.live_values["config"].set(str(config))
    calls = []
    ui.run = lambda args, on_success=None: calls.append((args, on_success))
    monkeypatch.setattr(windows_launcher.messagebox, "showinfo", lambda *args: None)
    ui.start_group_live()
    args, callback = calls[0]
    assert args[0] == "group-live"
    assert args[args.index("--animals") + 1] == "3"
    assert args[args.index("--cage-id") + 1] == "C1"
    session = tmp_path / "group_1"
    session.mkdir()
    (session / "group_session.json").write_text("{}", encoding="utf-8")
    callback()
    assert calls[1][0] == ["group-summary", str(tmp_path)]


def test_start_ai_uses_cpu_sampling_and_updates_separate_summary(tmp_path, monkeypatch):
    ui = launcher(tmp_path)
    ui.group_values["animals"].set("5")
    config = tmp_path / "layout.json"
    config.write_text("{}", encoding="utf-8")
    ui.live_values["config"].set(str(config))
    executable = tmp_path / ".ai_venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(windows_launcher, "ROOT", tmp_path)
    monkeypatch.setattr(windows_launcher.messagebox, "showinfo", lambda *args: None)
    calls = []
    ui.run = lambda args, on_success=None: calls.append((args, on_success))
    ui.start_group_ai()
    args, callback = calls[0]
    assert args[0] == "group-ai-live" and args[args.index("--interval") + 1] == "5.0"
    session = tmp_path / "ai_group_1"
    session.mkdir()
    (session / "ai_group_session.json").write_text("{}", encoding="utf-8")
    callback()
    assert calls[1][0] == ["group-ai-summary", str(tmp_path)]

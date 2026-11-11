"""Interactive setup: draw arenas and zones on a camera/video frame, calibrate px → cm, save config.

Mouse: left click = add point, right click = undo. Keys: Enter = finish polygon, Backspace = undo,
r = restart this shape, Esc = skip an optional zone, q = quit.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import draw
from .arena import Arena, Zone
from .config import new_config, save_config
from .geometry import Calibration, Circle, Polygon, grown_hull, order_corners
from .templates import Paradigm, paradigm

WINDOW = "behavtrack setup"
MAX_W, MAX_H = 1280, 800
KIND_KO = {"rect": "꼭짓점 4개 클릭", "circle": "중심 클릭 → 가장자리 클릭", "polygon": "꼭짓점들 클릭 후 Enter",
           "line": "양 끝점 2개 클릭"}


class Aborted(Exception):
    pass


def build_arena(par: Paradigm, name: str, shape, cal: Calibration, zones: list[Zone], params: dict,
                margin_cm: float = 3.0) -> Arena:
    """Assemble an arena: mazes use the hull of their zones; templates add automatic zones."""
    if par.arena_kind == "hull":
        if not zones:
            raise SystemExit("미로 실험은 구역을 하나 이상 그려야 합니다.")
        shape = grown_hull([z.shape for z in zones], margin_cm * cal.scale if cal.calibrated else 20.0)
    arena = Arena(name, shape, cal, list(zones))
    if par.auto_zones:
        arena.zones += par.auto_zones(arena, params)
    return arena


def render_preview(frame, arenas) -> np.ndarray:
    img = frame.copy()
    draw.layout(img, arenas)
    rows = []
    for a in arenas:
        c = a.calibration
        rows.append(f"{a.name}: {c.scale:.2f} px/cm" if c.calibrated else f"{a.name}: NOT calibrated (px units)")
    draw.lines(img, rows, origin=(10, img.shape[0] - 14 - 26 * (len(rows) - 1)))
    return img


class Drawer:
    def __init__(self, frame, frame_provider=None):
        self.frame = frame
        self.frame_provider = frame_provider
        h, w = frame.shape[:2]
        self.s = min(1.0, MAX_W / w, MAX_H / h)
        self.points: list[tuple[float, float]] = []
        self.cursor = None
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._mouse)

    def _mouse(self, event, x, y, flags, param):
        p = (x / self.s, y / self.s)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(p)
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
        elif event == cv2.EVENT_MOUSEMOVE:
            self.cursor = p

    def _show(self, img, rows):
        if self.s != 1.0:
            img = cv2.resize(img, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        draw.lines(img, rows)
        cv2.imshow(WINDOW, img)

    def _refresh_frame(self):
        if self.frame_provider is None:
            return
        ok, frame = self.frame_provider()
        if not ok or frame is None:
            raise SystemExit("카메라 영상이 끊어졌습니다. 연결을 확인하고 구역 설정을 다시 시작하세요.")
        if frame.shape != self.frame.shape:
            raise SystemExit("카메라 영상 크기가 바뀌었습니다. 구역 설정을 다시 시작하세요.")
        self.frame = frame

    def _render(self, done, kind, prompt):
        self._refresh_frame()
        img = self.frame.copy()
        for i, (name, shape) in enumerate(done):
            color = draw.ZONE_COLORS[i % len(draw.ZONE_COLORS)]
            shape.draw(img, color, 2)
            c = shape.centroid()
            draw.text(img, name, (c[0] - 20, c[1]), 0.5, color)
        pts = [tuple(int(v) for v in p) for p in self.points]
        cur = tuple(int(v) for v in self.cursor) if self.cursor else None
        yellow = (0, 255, 255)
        for p in pts:
            cv2.circle(img, p, 4, yellow, -1)
        if kind == "circle" and len(pts) == 1 and cur:
            cv2.circle(img, pts[0], int(math.dist(pts[0], cur)), yellow, 1, cv2.LINE_AA)
        elif pts:
            chain = pts + ([cur] if cur else [])
            cv2.polylines(img, [np.array(chain, np.int32)], kind != "line" and len(chain) > 2, yellow, 1, cv2.LINE_AA)
        self._show(img, [prompt, "right-click/Backspace: undo  r: restart  Esc: skip (optional)  q: quit"])

    def collect(self, kind: str, prompt: str, done, optional: bool = False):
        self.points = []
        need = {"rect": 4, "circle": 2, "line": 2}.get(kind)
        while True:
            self._render(done, kind, prompt)
            key = cv2.waitKey(20) & 0xFF
            if need and len(self.points) >= need:
                break
            if key in (13, 10) and kind == "polygon" and len(self.points) >= 3:
                break
            if key in (8, 127) and self.points:
                self.points.pop()
            elif key == ord("r"):
                self.points = []
            elif key == 27 and optional:
                return None
            elif key == ord("q"):
                raise Aborted
        pts = self.points[:need] if need else list(self.points)
        if kind == "rect":
            return Polygon(order_corners(pts))
        if kind == "circle":
            return Circle(pts[0], math.dist(pts[0], pts[1]))
        if kind == "line":
            return pts
        return Polygon(pts)

    def confirm(self, arenas) -> bool:
        while True:
            self._refresh_frame()
            self._show(render_preview(self.frame, arenas), ["Check the layout.  Enter/s: save   r: redo everything   q: quit"])
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10, ord("s")):
                return True
            if key == ord("r"):
                return False
            if key == ord("q"):
                raise Aborted


def _custom_zones(drawer: Drawer, name: str, done) -> list[Zone]:
    zones = []
    print("  구역을 추가합니다. (터미널에 입력 → 영상 창에서 그리기)")
    while True:
        zname = input("  구역 이름 (그만하려면 그냥 Enter): ").strip()
        if not zname:
            return zones
        kind = {"p": "polygon", "c": "circle"}.get(input("  모양 [r]사각형 / [p]다각형 / [c]원 (기본 r): ").strip()[:1], "rect")
        print(f"    → {KIND_KO[kind]}")
        shape = drawer.collect(kind, f"[{name}] {zname}: {kind}", done)
        zones.append(Zone(zname, shape))
        done.append((zname, shape))


def run_setup(frame, paradigm_key: str, out_path, n_arenas: int = 1, arena_size=None, arena_diameter=None,
              scale_length=None, species: str = "mouse", color: str = "dark", arena_shape: str = "rect",
              arena_names: list[str] | None = None, food_zone: bool = False, frame_provider=None) -> dict:
    par = paradigm(paradigm_key)
    cfg = new_config(paradigm_key, species, color)
    kind = par.arena_kind or arena_shape
    if kind == "rect" and not (arena_size or scale_length):
        print("⚠ --arena-size (예: 40x40) 가 없어 cm 보정을 할 수 없습니다. 결과가 픽셀 단위가 됩니다.")
    if kind == "circle" and not (arena_diameter or scale_length):
        print("⚠ --arena-diameter 가 없어 cm 보정을 할 수 없습니다. 결과가 픽셀 단위가 됩니다.")
    if kind in ("hull", "polygon") and not scale_length:
        print("⚠ --scale-length (화면에서 클릭할 기준 길이, cm) 가 없어 결과가 픽셀 단위가 됩니다.")
    print(f"\n[{par.title}] 영상 창에서 그려주세요. (영어 안내가 창 위쪽에 표시됩니다)")
    drawer = Drawer(frame, frame_provider)
    try:
        while True:
            arenas, done = [], []
            for i in range(n_arenas):
                name = (arena_names[i] if arena_names and i < len(arena_names)
                        else ("A" if n_arenas == 1 else f"A{i + 1}"))
                shape = None
                if kind in ("rect", "circle", "polygon"):
                    print(f"\n[{name}] 아레나 바깥 경계: {KIND_KO[kind]}")
                    prompt = {"rect": "ARENA: click its 4 corners", "circle": "ARENA: click center, then edge",
                              "polygon": "ARENA: click corners, Enter to finish"}[kind]
                    shape = drawer.collect(kind, f"[{name}] {prompt}", done)
                    done.append((name, shape))
                zones = []
                for spec in par.zones:
                    print(f"  - {spec.help_ko}: {KIND_KO[spec.kind]}" + (" (없으면 Esc)" if spec.optional else ""))
                    z = drawer.collect(spec.kind, f"[{name}] {spec.prompt}", done, spec.optional)
                    if z is not None:
                        zones.append(Zone(spec.name, z))
                        done.append((spec.name, z))
                if food_zone:
                    if shape is None:
                        raise SystemExit("사료 구역 지정에는 직접 그린 관찰 영역이 필요합니다.")
                    print("  사료 접근 구역: 경계를 차례로 클릭하고 Enter를 누르세요 (최소 3점).")
                    while True:
                        food = drawer.collect("polygon", f"[{name}] FOOD ZONE: click corners, then Enter", done)
                        if food.area_px() > 1 and np.all(shape.boundary_distance(food.outline()) <= 1e-6):
                            break
                        print("  사료 구역은 면적이 있어야 하며 관찰 영역 안에 그려야 합니다. 다시 그려주세요.")
                    zones.append(Zone("food", food))
                    done.append(("food", food))
                if par.key == "custom":
                    zones += _custom_zones(drawer, name, done)
                cal = Calibration()
                if kind == "rect" and arena_size:
                    cal = Calibration.from_rect(shape.outline(), *arena_size)
                elif kind == "circle" and arena_diameter:
                    cal = Calibration.from_circle(shape, arena_diameter)
                elif scale_length:
                    print(f"  - 보정: 실제 길이가 {scale_length:g} cm 인 두 점 클릭 (예: 팔 길이, 벽 길이)")
                    p0, p1 = drawer.collect("line", f"[{name}] SCALE: click both ends of a {scale_length:g} cm distance", done)
                    cal = Calibration.from_line(p0, p1, scale_length)
                arena = build_arena(par, name, shape, cal, zones, cfg["params"], cfg["tracking"]["arena_margin_cm"])
                arenas.append(arena)
                if kind == "hull":
                    done.append((name, arena.shape))
                done += [(z.name, z.shape) for z in arena.zones if z.auto]
            if drawer.confirm(arenas):
                break
            print("처음부터 다시 그립니다.")
    except Aborted:
        raise SystemExit("설정을 취소했습니다.") from None
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    cfg["arenas"] = [a.to_dict() for a in arenas]
    out_path = Path(out_path)
    save_config(cfg, out_path)
    cv2.imwrite(str(out_path.with_suffix(".preview.png")), render_preview(drawer.frame, arenas))
    cv2.imwrite(str(out_path.with_suffix(".reference.png")), drawer.frame)
    return cfg

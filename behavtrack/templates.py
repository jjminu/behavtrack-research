"""Built-in experiment paradigms: what to draw during setup, auto-generated zones, default parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .arena import Arena, Zone
from .geometry import Circle, Polygon, order_corners, rect_subregion


@dataclass
class ZoneSpec:
    name: str
    kind: str  # "rect" (4 corners), "polygon", "circle"
    prompt: str  # shown on the video window (OpenCV can only draw ASCII)
    help_ko: str  # printed in the terminal
    optional: bool = False


@dataclass
class Paradigm:
    key: str
    title: str
    arena_kind: str | None  # "rect", "circle", "hull" (arena = hull of the drawn zones), None = ask
    zones: list[ZoneSpec] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    auto_zones: Callable[[Arena, dict], list[Zone]] | None = None
    description: str = ""


def _open_field_zones(arena: Arena, params: dict) -> list[Zone]:
    m = (1 - float(params.get("center_fraction", 0.5))) / 2
    return [Zone("center", rect_subregion(arena.shape.outline(), m, m, 1 - m, 1 - m), auto=True)]


def _three_chamber_zones(arena: Arena, params: dict) -> list[Zone]:
    corners = order_corners(arena.shape.outline())
    cm = arena.calibration.to_cm(corners) if arena.calibration.calibrated else corners
    horizontal = math.dist(cm[0], cm[1]) >= math.dist(cm[1], cm[2])
    names = ["chamber_A", "chamber_center", "chamber_B"]
    zones = []
    for i, name in enumerate(names):
        a, b = i / 3, (i + 1) / 3
        poly = rect_subregion(corners, a, 0, b, 1) if horizontal else rect_subregion(corners, 0, a, 1, b)
        zones.append(Zone(name, poly, auto=True))
    return zones


QUADRANTS = {"Q_NE": (-90, 0), "Q_SE": (0, 90), "Q_SW": (90, 180), "Q_NW": (180, 270)}


def _quadrant_zones(arena: Arena, params: dict) -> list[Zone]:
    """Four pool quadrants; north is the top of the image."""
    if not isinstance(arena.shape, Circle):
        return []
    c, r = arena.shape.center, arena.shape.radius
    zones = []
    for name, (a0, a1) in QUADRANTS.items():
        ang = np.radians(np.linspace(a0, a1, 25))
        arc = c + r * np.column_stack([np.cos(ang), np.sin(ang)])
        zones.append(Zone(name, Polygon(np.vstack([c, arc])), auto=True))
    return zones


def _arm(name: str, what: str, ko: str) -> ZoneSpec:
    return ZoneSpec(name, "polygon", f"{what}: click corners, Enter to finish", ko)


PARADIGMS: dict[str, Paradigm] = {
    p.key: p
    for p in [
        Paradigm(
            "open_field", "오픈필드 (Open field)", "rect",
            params={"center_fraction": 0.5},
            auto_zones=_open_field_zones,
            description="이동성·불안: 이동거리, 속도, 중앙 체류시간/진입, 가장자리 선호(thigmotaxis)",
        ),
        Paradigm(
            "epm", "고가십자미로 (Elevated plus maze)", "hull",
            zones=[
                _arm("open_1", "OPEN arm 1", "열린 팔 1"),
                _arm("open_2", "OPEN arm 2", "열린 팔 2"),
                _arm("closed_1", "CLOSED arm 1", "닫힌 팔 1"),
                _arm("closed_2", "CLOSED arm 2", "닫힌 팔 2"),
                ZoneSpec("center", "rect", "CENTER square: click 4 corners", "가운데 정사각형 (꼭짓점 4개)"),
            ],
            description="불안: 열린 팔 체류시간 %, 열린 팔 진입 %, 총 진입수, (포즈 사용 시) head dip",
        ),
        Paradigm(
            "ezm", "고가 제로미로 (Elevated zero maze)", "hull",
            zones=[
                _arm("open_1", "OPEN sector 1", "열린 구간 1"),
                _arm("open_2", "OPEN sector 2", "열린 구간 2"),
                _arm("closed_1", "CLOSED sector 1", "닫힌 구간 1"),
                _arm("closed_2", "CLOSED sector 2", "닫힌 구간 2"),
            ],
            description="불안: 열린 구간 체류시간 %, 진입수",
        ),
        Paradigm(
            "y_maze", "Y자 미로 자발적 교대 (Y-maze)", "hull",
            zones=[
                _arm("arm_A", "ARM A", "팔 A"),
                _arm("arm_B", "ARM B", "팔 B"),
                _arm("arm_C", "ARM C", "팔 C"),
                ZoneSpec("center", "polygon", "CENTER triangle: click corners, Enter", "가운데 삼각형", optional=True),
            ],
            params={"count_start_arm": False},
            description="작업기억: 팔 진입 순서, 자발적 교대율 %, 같은 팔 재진입",
        ),
        Paradigm(
            "nor", "새 물체 인식 (Novel object recognition)", "rect",
            zones=[
                ZoneSpec("object_1", "circle", "OBJECT 1: click center, then edge", "물체 1 (중심 → 가장자리)"),
                ZoneSpec("object_2", "circle", "OBJECT 2: click center, then edge", "물체 2 (중심 → 가장자리)"),
            ],
            params={"novel_object": None, "explore_cm": 2.0, "centroid_explore_cm": 4.0,
                    "exclude_on_object": True, "min_explore_s": 0.0, "criterion_s": None},
            description="인식기억: 물체별 탐색시간, 변별지수(DI), 선호도 % (코 키포인트가 있으면 코 기준)",
        ),
        Paradigm(
            "light_dark", "명암상자 (Light-dark box)", "rect",
            zones=[ZoneSpec("dark", "rect", "DARK compartment: click 4 corners", "어두운 칸 (꼭짓점 4개)")],
            params={"undetected_as": "unknown"},
            description="불안: 밝은 칸 체류시간, 칸 이동 횟수, 어두운 칸 진입 잠복기",
        ),
        Paradigm(
            "three_chamber", "3구획 사회성 (Three-chamber)", "rect",
            zones=[
                ZoneSpec("cup_A", "circle", "CUP in chamber A (left/top): center, then edge", "A칸(왼쪽/위) 컵"),
                ZoneSpec("cup_B", "circle", "CUP in chamber B (right/bottom): center, then edge", "B칸(오른쪽/아래) 컵"),
            ],
            params={"social_side": None, "interaction_cm": 3.0, "explore_cm": 2.0},
            auto_zones=_three_chamber_zones,
            description="사회성/사회적 새로움: 칸별 체류시간, 컵 탐색시간, 선호 지수",
        ),
        Paradigm(
            "cpp", "조건화 장소선호 (CPP/CPA)", "rect",
            zones=[
                ZoneSpec("chamber_A", "rect", "CHAMBER A: click 4 corners", "A칸 (꼭짓점 4개)"),
                ZoneSpec("chamber_B", "rect", "CHAMBER B: click 4 corners", "B칸 (꼭짓점 4개)"),
                ZoneSpec("neutral", "rect", "NEUTRAL chamber (Esc if none)", "중립칸 (없으면 Esc)", optional=True),
            ],
            params={"paired_chamber": None},
            description="보상/혐오: 칸별 체류시간, 선호 점수(짝지은 칸 − 반대 칸)",
        ),
        Paradigm(
            "mwm", "모리스 수중미로 (Morris water maze)", "circle",
            zones=[ZoneSpec("platform", "circle", "PLATFORM: click center, then edge", "플랫폼 (중심 → 가장자리)")],
            params={"target_quadrant": "auto", "wall_zone_cm": 10.0, "approach_cm": 0.0},
            auto_zones=_quadrant_zones,
            description="공간기억: 도피 잠복기, 경로 길이, 목표 사분면 %, 플랫폼 통과 횟수, 벽 따라 수영",
        ),
        Paradigm(
            "barnes", "반스 미로 (Barnes maze)", "circle",
            zones=[ZoneSpec("target", "circle", "TARGET (escape) hole: center, then edge", "탈출 구멍 (중심 → 가장자리)")],
            params={"target_quadrant": "auto", "approach_cm": 3.0},
            auto_zones=_quadrant_zones,
            description="공간기억: 1차 잠복기, 1차 경로 길이, 목표 사분면 %",
        ),
        Paradigm(
            "fear_conditioning", "공포 조건화 (Freezing)", "rect",
            params={"epochs": [], "tone_onsets": [], "tone_duration_s": 20.0},
            description="공포기억: 전체/구간별(기준선·톤) freezing %, freezing 횟수",
        ),
        Paradigm(
            "custom", "사용자 정의", None,
            description="원하는 구역을 직접 그려서 구역별 체류시간·진입·이동거리 등 공통 지표만 계산",
        ),
    ]
}


def paradigm(key: str) -> Paradigm:
    try:
        return PARADIGMS[key]
    except KeyError:
        raise SystemExit(f"알 수 없는 실험 종류: {key} (가능: {', '.join(PARADIGMS)})") from None

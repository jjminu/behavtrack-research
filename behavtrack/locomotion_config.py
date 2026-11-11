"""Lightweight validation shared by the capture client and analysis server."""
from __future__ import annotations
import math
import cv2
import numpy as np

VERSION = 'cage_centroid_locomotion_v1'
DEFAULTS = dict(interval_s=.2, time_tolerance_s=.04, min_score=.6, max_iou=.15,
                ambiguity_ratio=.75, max_speed_cm_s=100., max_step_diagonal=.12,
                deadband_cm=0., bin_seconds=60., white_threshold=180.,max_saturation=65.,
                min_area_fraction=.0015,max_area_fraction=.04,opening=5.)

def calibration(value):
    if value is None:return None
    if not isinstance(value,dict):raise ValueError('바닥 보정 정보를 확인하세요.')
    pts=np.asarray(value.get('points'),dtype=float)
    if pts.shape!=(4,2) or not np.isfinite(pts).all() or np.any(pts<0) or np.any(pts>1):
        raise ValueError('바닥 모서리 네 점을 왼쪽 위→오른쪽 위→오른쪽 아래→왼쪽 아래 순서로 지정하세요.')
    edges=np.roll(pts,-1,axis=0)-pts
    cross=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
    if not np.all(cross>1e-5) or cv2.contourArea(pts.astype(np.float32))<.005:
        raise ValueError('바닥 네 점은 겹치지 않는 시계 방향 사각형이어야 합니다.')
    width,height=float(value.get('width_cm',0)),float(value.get('height_cm',0))
    if not all(math.isfinite(x) and 1<=x<=300 for x in (width,height)):
        raise ValueError('실측한 바닥 가로·세로 길이를 각각 1~300 cm로 입력하세요.')
    if value.get('planar_confirmed') is not True:raise ValueError('고정된 상부 카메라와 평평한 바닥을 확인해 주세요.')
    return dict(points=pts.tolist(),width_cm=width,height_cm=height,planar_confirmed=True,
                lengths_approximate=value.get('lengths_approximate',False) is True)

def settings(values):
    out={k:float(values.get(k,v)) for k,v in DEFAULTS.items()}
    if not all(math.isfinite(v) for v in out.values()):raise ValueError('추적 설정은 유한한 수여야 합니다.')
    if not (.1<=out['interval_s']<=.5 and 0<out['time_tolerance_s']<out['interval_s']/2
            and .1<=out['min_score']<=1 and 0<=out['max_iou']<=.5 and .1<=out['ambiguity_ratio']<1
            and 1<=out['max_speed_cm_s']<=500 and .001<=out['max_step_diagonal']<=.5
            and 0<=out['deadband_cm']<=2 and 1<=out['bin_seconds']<=3600
            and 40<=out['white_threshold']<=250 and 0<=out['max_saturation']<=255
            and .0001<=out['min_area_fraction']<out['max_area_fraction']<=.2 and out['opening'] in (1,3,5,7,9)):
        raise ValueError('이동거리 추적 설정 범위를 확인하세요.')
    detector=values.get('detector','dlc')
    if detector not in ('dlc','bright_foreground'):raise ValueError('마우스 검출 방식을 선택하세요.')
    out.update(method=VERSION,calibration=calibration(values.get('calibration')),detector=detector)
    return out

def requested(values):
    """Validate optional automatic locomotion before the camera starts."""
    v=values.get('locomotion')
    if v is None or v is False:return None
    if not isinstance(v,dict):raise ValueError('자동 이동거리 분석 설정을 확인하세요.')
    return settings(v)

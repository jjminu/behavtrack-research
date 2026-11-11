# 선택 기능용 모델

기본 촬영, DeepLabCut CSV 가져오기, 명암 기반 이동거리 데모는 모델 다운로드 없이 실행됩니다. SLEAP HDF5 가져오기는 `pose` 추가 의존성이 필요합니다.

DeepLabCut 기반 검출을 사용하려면 Windows에서 별도의 `.ai_venv`를 만든 뒤 DeepLabCut 3.0.2를 설치합니다.

```powershell
py -3.11 -m venv .ai_venv
.\.ai_venv\Scripts\python.exe -m pip install "deeplabcut==3.0.2"
```

[SuperAnimal-TopViewMouse 모델 배포처](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)에서 모델 제공자의 이용 조건을 확인합니다. 검출 가중치 `detector.pt`는 `models/topviewmouse/detector.pt`에 둡니다. 기존 포즈 기능을 사용하려면 배포처의 `pose_model.pth`를 `models/topviewmouse/pose.pt`에, [TorchVision MobileNet V3 Large 가중치](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth)를 `models/torch/hub/checkpoints/mobilenet_v3_large-8738ca79.pth`에 둡니다.

[모델 카드와 이용 조건](topviewmouse/MODEL_CARD_LICENSE.md) · [TorchVision 라이선스 고지](torch/TORCHVISION_LICENSE.txt)

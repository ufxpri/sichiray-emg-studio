# Sichiray EMG Studio

**Sichiray EMG PRO 8-Channel Armband Workbench & EMG-to-Pose Learner**

Sichiray EMG PRO 8채널 암밴드를 위한 멀티 윈도우 작업대입니다. 근전도(EMG) 신호를 실시간으로 받아 시각화하고 노이즈를 필터링합니다. 웹캠 기반 3D 손 인식 모델(WiLoR)을 정답(Ground Truth)으로 삼아 **EMG -> 손 자세 예측 모델**을 현장에서 바로 학습할 수 있습니다.

## 주요 기능 (Features)

- **실시간 신호 모니터링:** 8채널 EMG 원시 데이터(HEX/ASCII)와 IMU 센서 데이터 스트리밍
- **신호 처리와 노이즈 진단:** 실시간 롤링 스펙트로그램, 채널별 노이즈 진단, 사용자 지정 필터 체인(고역, 저역, 전원 노치 등)
- **웹캠 3D 손 추적:** GPU 기반 WiLoR 모델로 오른손 21관절을 실시간 3D 추적
- **현장 학습 (On-the-spot Learning):** 카메라가 인식한 손 모양을 정답으로 삼아, EMG 신호만으로 손 형태를 복원하는 모델을 수집, 학습, 검증

## 시스템 요구 사항 (Prerequisites)

- **OS:** Windows 10 / 11
- **Hardware:** Sichiray EMG PRO 8채널 암밴드, USB 수신기(CH340), 웹캠
- **Software:**
  - 메인 스튜디오: Python 3.12
  - 3D 손 인식 모듈: Python 3.10, NVIDIA GPU (선택 사항)

## 설치 (Installation)

### 1. 기본 패키지 설치

```bash
git clone https://github.com/ufxpri/sichiray-emg-studio.git
cd sichiray-emg-studio
python -m pip install -r requirements.txt
```

### 2. 카메라 손 인식(WiLoR) 모듈 설치 (선택 사항)

카메라 기반 손 인식에는 PyTorch(NVIDIA GPU) 기반의 별도 Python 3.10 환경이 필요합니다.

- 설치 경로: `%USERPROFILE%\.emg_hand` (약 7.5 GB)
- 패키지 관리자 `uv`가 필요합니다.

```bash
mkdir -p ~/.emg_hand && cd ~/.emg_hand
uv venv --python 3.10 venv

uv pip install --python venv/Scripts/python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python venv/Scripts/python.exe pip setuptools wheel "numpy<2" scipy dill
uv pip install --python venv/Scripts/python.exe --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
uv pip install --python venv/Scripts/python.exe "numpy<2" smplx==0.1.28 timm einops ultralytics==8.1.34 opencv-python huggingface_hub scikit-image roma
uv pip install --python venv/Scripts/python.exe --no-deps "git+https://github.com/warmshao/WiLoR-mini"
```

> **참고:** 손 인식 모델 가중치(약 2.5 GB)는 첫 실행 때 자동으로 내려받습니다. 다른 위치의 Python을 쓰려면 환경 변수 `EMG_HAND_PY`로 지정합니다.

## 사용 방법 (Usage)

스튜디오를 실행하면 런처가 열리고, 연결할 포트를 고를 수 있습니다.

```bash
python emg_studio.py
```

**CLI 옵션:**

```bash
# 포트와 웹캠을 지정하고 모든 창 열기
python emg_studio.py --port COM4 --camera 0 --open all

# 장치 없이 데모(합성 신호) 모드로 실행
python emg_studio.py --demo hex --open all

# 테스트 실행
python -m pytest tests
```

### 모델 학습 워크플로우

1. 카메라 창에서 **[시작]** 클릭
2. 3D 창에서 **[수집 시작]** 클릭 후 1~2분간 손가락을 다양하게 움직임
3. **[수집 중지]** -> **[학습]** -> **[EMG 예측]** 활성화

학습 결과에는 "평균 손" 기준 대비 개선폭과 우연 수준 판정(p값)이 함께 표시됩니다.

## 하드웨어 참고 사항 (Hardware Specs & Known Issues)

<img src="assets/armband_channels.jpg" alt="채널 번호 라벨을 붙인 8채널 밴드. 5번 셀에 USB-C 포트와 녹색 LED가 있다" width="480">

- **모드 전환:** 전원을 켜면 `HEX` 모드(8-bit, 500 Hz)로 동작합니다. 버튼을 2회 누르면 `ASCII` 모드(12-bit, 67 Hz), 1회 누르면 `HEX` 모드로 바뀝니다.
- **통신:** Bluetooth 5.0 -> USB 수신기(CH340, VID 1A86 / PID 7523), 115200 baud 8N1
- **HEX 프레임 (98 B, 50 fps):** `AA AA 5F | ts(4, BE ms) | Acc(3) Gyro(3) Angle(3) | EMG 10샘플 x 8ch | 배터리 | 55`
- **센서 구조:** 셀 8개로 구성됩니다. CH5(USB-C 포트와 녹색 LED가 있는 셀)를 기준으로 번호가 역순(CH4, CH3, ...)으로 매겨지며, CH1과 CH8은 서로 이웃합니다. 착용할 때 밴드가 돌아간 만큼은 필터 창의 **채널 회전**으로 보정합니다.
- **알려진 이슈 (CH5 노이즈):**
  - CH5 셀 안의 메인 보드와 아날로그 보드가 절연 없이 겹쳐 있어 **광대역 잡음, DC 오프셋, 신호 포화**가 생길 수 있습니다.
  - *해결 방법:* 기기를 분해해 두 보드 사이에 절연 테이프를 붙이면 노이즈가 정상 범위로 크게 줄어듭니다.

## 프로젝트 구조 (Directory Structure)

```text
sichiray-emg-studio/
├── emg_studio.py          # 메인 실행 진입점
├── worker/                # 별도 프로세스 워커 (hand_worker.py - WiLoR 손 인식)
├── studio/
│   ├── config.py          # 경로, 포트, 환경 변수 설정
│   ├── device/            # 하드웨어 통신 프로토콜과 데이터 파서
│   ├── core/              # 링 버퍼, 필터 파이프라인, 노이즈 분석
│   ├── hand/              # 카메라 링크, 스켈레톤, MANO 손 모델
│   ├── pose/              # 특징 추출, 데이터셋 수집, 학습(Ridge/MLP)과 평가
│   └── ui/                # PySide6 UI 컴포넌트와 OpenGL 뷰어
├── tests/                 # 회귀 테스트
├── assets/                # 문서용 이미지
└── data/                  # 런타임 설정, 수집 데이터, 학습 모델 (gitignore)
```

## 라이선스 (License)

이 프로젝트의 코드는 [MIT License](LICENSE)를 따릅니다.

웹캠 손 인식에 쓰는 `WiLoR` / `WiLoR-mini` 가중치와 `MANO` 손 모델은 이 저장소에 포함되어 있지 않으며, 비상업적 연구 목적으로만 사용할 수 있습니다. 각 원작자의 라이선스를 따르고, 상업적으로 이용하려면 원작자의 허가가 필요합니다.

# VAD Preprocess

SGVAD(Stochastic Gates Voice Activity Detection) 모델을 활용하여 오디오 파일에서 음성 구간을 감지하고 처리하는 도구입니다.
sgvad: arXiv:2210.16022

## 개요

이 리포지토리는 VAD(Voice Activity Detection) 시스템을 쉽게 사용할 수 있도록 구현한 도구 모음입니다. SGVAD 모델을 기반으로 오디오 파일에서 음성 구간을 감지하고, 다양한 처리 및 시각화 기능을 제공합니다.

**주요 변경사항**: SGVAD의 추론 방식이 수정되어 기존 코드와 호환성 문제가 있을 수 있습니다. 본 리포지토리의 코드를 참조하여 최신 방식으로 사용하시기 바랍니다.

## 환경 설정

### 요구사항

- Python 3.10
- UV 패키지 관리자
- CUDA 지원 GPU (선택사항, CPU 버전 사용 가능)

### 설치 방법

1. 리포지토리 클론:
```bash
git clone https://github.com/LEE5J/vad_preprocess
cd vad_preprocess
```

2. UV를 사용하여 가상환경 설정:
```bash
# UV 설치 (이미 설치되어 있지 않은 경우)
curl -sSf https://install.determinate.systems/uv | sh

# 가상환경 생성 및 활성화
uv venv -p python3.10
source .venv/bin/activate  # Linux/macOS
# 또는
.venv\Scripts\activate  # Windows

# uv.lock 파일에서 의존성 설치
uv pip install -r uv.lock
```

참고: requirements.txt 파일도 있지만, 일부 패키지는 pip만으로 설치되지 않을 수 있습니다. UV를 사용하는 것이 권장됩니다.

## 프로젝트 구조 및 기능

### 핵심 모듈

- **sgvad.py**: 기본 SGVAD 구현. 스무딩과 MFCC 추출 기능을 포함합니다. 싱글스레드 작업에 적합합니다.
- **sgvad_cpu.py**: CPU에서 최적화된 버전으로, GPU가 없는 환경에서 사용할 수 있습니다.
- **sgvad_jit.py**: PyTorch Script 파일(.pt)을 생성하는 도구입니다.
- **sgvad_jit_test.py**: 생성된 스크립트 모델이 원래 모델과 동일한 결과를 내는지 확인합니다.
- **sgvad_jit_example.py**: TorchScript 모델 사용 예제를 제공합니다.
- **parse_textgrid.py**: TextGrid 형식의 라벨링 데이터를 읽고 처리하는 커스텀 구현입니다.
- **is_noise_acceptable.py**: 오디오 파일의 SNR(신호 대 잡음비) 성능을 평가합니다.

### 유틸리티

- **snr_sgvad_test.py**: 다양한 SNR 조건에서의 SGVAD 성능을 테스트합니다.
- **vad_visualize_cpu.py** / **vad_visualize_gpu.py**: 음성 감지 결과를 시각화하는 도구입니다.
- **audio_segment_processor.py**: 오디오 세그먼트 처리를 위한 유틸리티입니다.

## 사용 방법

### SGVAD 기본 사용법

```python
from sgvad import SGVAD

# SGVAD 모델 초기화
model = SGVAD()

# 오디오 파일 리스트 준비
audio_files = ["path/to/audio1.wav", "path/to/audio2.wav", ...]

# VAD 예측 수행
vad_results = model.predict(audio_files)

# 결과 처리 (주의: 마지막 파일이 처리되지 않을 수 있음)
for audio_path, scores in zip(audio_files, vad_results):
    if scores:  # 빈 리스트가 아니면
        print(f"{audio_path}: {len(scores)} 프레임, 평균 점수: {sum(scores)/len(scores):.2f}")
```

### CPU 버전 사용법

```python
from sgvad_cpu import SGVAD

# CPU 버전 초기화
model = SGVAD(cfg_path="./cfg.yaml")

# 오디오 파일 리스트 준비 및 예측
audio_files = ["path/to/audio1.wav", "path/to/audio2.wav", ...]
vad_results = model.predict(audio_files, smooth=15, num_workers=4)  # 워커 수 지정 가능
```

### SNR 평가 

SNR 은 노이스와 신호비를 의미한다.
여기서는 퍼센타일 값 기준으로 15%와 95% 비율을 측정한다.
이는 노이즈와 신호를 각각 의미한다.
이 비율이 노이즈 상황일 경우 10배 정도 차이난다.


## 주요 특징 및 고려사항

1. **프레임 사이즈**: VAD는 10ms(0.01초) 단위의 고정 슬라이싱 윈도우를 사용합니다.

2. **성능 고려사항**:
   - 모델 추론 자체는 계산 비용이 크지 않으며 CPU로도 충분히 실행 가능합니다.
   - MFCC 추출 과정이 CPU 부하가 크며, GPU와 비교해 약 20배 정도 느릴 수 있습니다.
   - GPU 환경에서는 내부적으로 병렬 처리가 이루어지므로 현재 싱글스레드 방식을 채택했습니다.

3. **Whisper와의 통합**:
   - Whisper가 TextGrid 형식의 출력을 지원하므로 이를 활용한 라벨링 데이터 처리 기능이 포함되어 있습니다.
   - Whisper는 노이즈 환경에서도 비교적 우수한 성능을 보이나, 사람의 라벨링과 비교했을 때 약 80% 정확도 수준에 머무르는 한계가 있습니다.
   - Whisper 관련 시각화 도구는 추후 별도 코드로 분리될 예정입니다.

4. **배치 처리 고려사항**:
   - 대규모 상용 배치 처리를 위해서는 TorchScript(.pt) 방식이 권장됩니다.
   - 본 프로젝트는 주로 테스트 및 실험용 환경을 위해 개발되었습니다.

5. **TextGrid 파싱**:
   - Whisper로 생성된 TextGrid 파일은 표준 라이브러리로 처리 시 문제가 발생할 수 있어 사용자 정의 파서를 구현했습니다.

## 향후 계획

- 다양한 SGVAD 버전을 개발하여 사용 사례에 맞게 선택할 수 있도록 할 예정입니다.
- Whisper 통합 시각화 툴을 별도 코드로 분리할 계획입니다.
- 배치 처리 성능 최적화를 진행할 예정입니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

---

문의 사항이나 기여는 언제든지 환영합니다!
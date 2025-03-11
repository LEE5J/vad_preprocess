# VAD 시각화 도구 사용 설명서

## 개요

이 시각화 도구는 SGVAD(Stochastic Gates Voice Activity Detection) 모델을 활용하여 오디오 파일의 발화 부분을 감지하고 시각화 도구입니다. CPU 전용 모드와 GPU 가속 모드를 모두 지원하여 다양한 환경에서 사용할 수 있습니다.

## 주요 기능

- **고품질 시각화**: 오디오 파형과 발화 확률 점수를 보여주는 정적 이미지 또는 비디오 생성
- **멀티모드 지원**: CPU 또는 GPU 처리 방식 선택 가능
- **병렬 처리**: 다중 프로세스를 통한 대량 파일 처리
- **사용자 정의 매개변수**: 세그먼트 지속 시간, 스무딩, 프레임 레이트 등 조정 가능

## 설치 요구사항

### 사전 요구사항
- 기본적으로 ubuntu 22.04 에서 테스트 되었으며 CPU 버전은 맥과 윈도우에서도 지원될 수 있습니다.
- Python 3.7 이상
- PyTorch
- FFmpeg (GPU 가속 비디오 생성에 필요)
- Matplotlib
- NumPy
- SoundFile
- CUDA와 NVENC 지원 그래픽 카드 (GPU 모드)

### 설정

1. 필요한 모듈 설치:
```bash
pip install torch numpy soundfile librosa matplotlib tqdm omegaconf
```

2. FFmpeg 설치 (시스템 패키지 관리자 사용):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg
```

3. SGVAD 모델 파일(`sgvad.pth`)과 구성 파일(`cfg.yaml`)이 프로젝트 디렉토리에 있는지 확인하세요.

## 기본 사용법

### CPU 모드 (vad_visualize_cpu.py)

CPU 전용 모드는 Matplotlib를 사용하여 애니메이션을 생성하며, 그래픽 카드가 없는 환경에서도 작동합니다.

```bash
python vad_visualize_cpu.py --input "오디오_디렉토리/*.wav" --output "출력_디렉토리"
```

### GPU 모드 (vad_visualize_gpu.py)

GPU 가속 모드는 NVENC를 사용하여 빠른 비디오 인코딩을 제공합니다. CUDA와 NVENC를 지원하는 NVIDIA 그래픽 카드가 필요합니다.

```bash
python vad_visualize_gpu.py --input "오디오_디렉토리/*.wav" --output "출력_디렉토리"
```

## 매개변수 상세 정보

### 공통 매개변수 (CPU/GPU 모드 모두 적용)

- `--input`: 입력 오디오 파일 패턴 (예: "data/*.wav")
- `--output`: 결과 저장 디렉토리
- `--max_files`: (선택 사항) 처리할 최대 파일 수
- `--seg_duration`: 오디오 세그먼트 길이 (초 단위, 기본값: 30)
- `--merge_threshold`: 짧은 세그먼트 병합 임계값 (초 단위, 기본값: 10)
- `--smoothing_kernel`: 스무딩 커널 크기 (기본값: 21)
- `--fps`: 비디오 프레임 레이트 (기본값: 30)
- `--workers` 또는 `-w`: 병렬 처리에 사용할 프로세스 수

### GPU 모드 전용 매개변수

- `--static_plot`: 비디오 대신 정적 이미지로 시각화 결과 저장

## CPU와 GPU 모드 비교

### CPU 모드 (vad_visualize_cpu.py)
- **장점**: 그래픽 카드가 필요 없음, 모든 시스템 호환
- **단점**: 처리 속도가 느림, 특히 비디오 생성 시간이 오래 걸림 (100배 정도 차이납니다)
- **용도**: 고사양 그래픽 카드가 없는 환경, 서버 환경

### GPU 모드 (vad_visualize_gpu.py)
- **장점**: 빠른 비디오 인코딩, 대량 처리에 효율적
- **단점**: NVIDIA 그래픽 카드와 NVENC 지원 필요
- **용도**: 대용량 처리, 고품질 비디오 출력 필요 시
- **참고**: 소비자용 그래픽 카드는 기본적으로 동시 인코딩 세션 수가 제한되어 있으며, 이는 [nvidia-patch](https://github.com/keylase/nvidia-patch)를 통해 해제할 수 있습니다

## 단계별 튜토리얼

1. **오디오 파일 준비**:
   - WAV 파일을 디렉토리에 배치 (16kHz 샘플링 레이트 권장)

2. **출력 디렉토리 생성**:
   ```bash
   mkdir -p visualization_results
   ```

3. **CPU 모드로 실행**:
   ```bash
   python vad_visualize_cpu.py --input "your_audio/*.wav" --output "visualization_results" 
   ```

4. **GPU 모드로 실행**:
   ```bash
   python vad_visualize_gpu.py --input "your_audio/*.wav" --output "visualization_results"
   ```

5. **결과 확인**:
   - 정적 이미지: `visualization_results/{filename}/{filename}_plot.png`
   - 비디오: `visualization_results/{filename}/{filename}_visualization.mp4`

## 고급 사용 예시

### 더 작은 세그먼트 크기와 증가된 병합 임계값 설정

```bash
python vad_visualize_gpu.py --input "audio/*.wav" --output "results" --seg_duration 15 --merge_threshold 5
```

### 더 높은 프레임 레이트로 비디오 생성

```bash
python vad_visualize_gpu.py --input "audio/*.wav" --output "results" --fps 60
```

### 정적 이미지만 생성 (비디오 생성 건너뛰기)

```bash
python vad_visualize_gpu.py --input "audio/*.wav" --output "results" --static_plot
```

### 더 많은 병렬 프로세스로 처리 (CPU 모드)

```bash
python vad_visualize_cpu.py --input "audio/*.wav" --output "results" --workers 16
```

## 시각화 출력 설명

### 정적 플롯 구성
생성된 정적 이미지는 두 개의 서브플롯으로 구성됩니다:

1. **상단 서브플롯**: 오디오 파형과 감지된 발화 구간 (빨간색 배경)
2. **하단 서브플롯**: VAD 확률 점수, 60번째 및 90번째 퍼센타일 임계값

### 비디오 출력 특징
- 상단/하단 서브플롯 구성은 정적 플롯과 동일
- 빨간색 수직선으로 현재 재생 위치 표시
- 오디오가 동기화되어 재생

## 사용자 지정 및 팁

### 스무딩 조정

스무딩 커널 크기는 VAD 확률 점수의 노이즈를 줄이는 데 영향을 미칩니다:
- 기본값 21은 중간 정도의 스무딩을 제공합니다
- 더 작은 값(예: 11)은 더 세부적인 변화를 보존합니다
- 더 큰 값(예: 31)은 더 부드러운 결과를 제공합니다

```bash
python vad_visualize_gpu.py --input "audio/*.wav" --output "results" --smoothing_kernel 11
```

### GPU 최적화

소비자용 NVIDIA 그래픽 카드를 사용하는 경우:
- 기본적으로 최대 8개의 동시 인코딩 세션만 지원합니다
- 더 많은 세션을 활용하려면 [nvidia-patch](https://github.com/keylase/nvidia-patch)를 적용할 수 있습니다.
- 이렇게 적용할 경우 -w 옵션을 통해서 더 많은 스레드를 할당 해주어야 성능을 최대로 끌어 올릴 수 있습니다.
- 프로페셔널 GPU는 제한이 없거나 더 높은 제한을 가집니다
- 데이터 센터용 GPU 는 사용불가능합니다.

### 참고: A100과 같은 데이터 센터용 GPU

A100과 같은 데이터 센터용 GPU는 NVENC를 지원하지 않으므로 CPU 모드를 사용해야 합니다:

```bash
python vad_visualize_cpu.py --input "audio/*.wav" --output "results"
```

## 문제 해결

### GPU 모드 사용 시 오류 발생

1. NVIDIA 드라이버가 최신 버전인지 확인합니다 테스트 환경은 550.120 과 12.4 쿠다 버전을 사용하고 있습니다.
2. CPU 인코딩 버전을 고려하세요
3. NVENC 를 지원하고 ffmpeg 이 동작하는지를 점검하세요

### 메모리 부족 오류

1. 세그먼트 지속 시간을 줄입니다: `--seg_duration 15`
2. 한 번에 처리하는 스레드 수를 제한합니다: `-w 10`

### 긴 오디오 파일 처리 시 시간 초과

매우 긴 오디오 파일의 경우 세그먼트 지속 시간과 병합 임계값을 조정하세요:
```bash
python vad_visualize_gpu.py --input "long_files/*.wav" --output "results" --seg_duration 60 --merge_threshold 15
```

## 출력 구조

각 처리된 오디오 파일(`example.wav`)에 대한 출력은 다음과 같습니다:
```
results/
└── example/
    ├── example.wav_00_speech.wav      # 첫 번째 발화 세그먼트
    ├── example.wav_01_silence.wav     # 첫 번째 무음 세그먼트
    ├── example_plot.png               # 정적 시각화 이미지
    └── example_visualization.mp4      # 비디오 시각화 (생성된 경우)
```

## 라이선스

이 프로젝트는 비상업적 용도로만 제공됩니다.

# VAD Visualization Tool User Guide

## Overview

This visualization tool detects and visualizes speech segments in audio files using the SGVAD (Stochastic Gates Voice Activity Detection) model. It supports both CPU-only mode and GPU-accelerated mode for use in various environments.

## Key Features

- **High-quality visualization**: Generates static images or videos showing audio waveforms and speech probability scores
- **Multi-mode support**: Choose between CPU or GPU processing methods
- **Parallel processing**: Process large batches of files through multiple processes
- **Customizable parameters**: Adjust segment duration, smoothing, frame rate, and more

## Installation Requirements

### Prerequisites
- Primarily tested on Ubuntu 22.04, though CPU version may also work on Mac and Windows
- Python 3.7 or higher
- PyTorch
- FFmpeg (required for GPU-accelerated video generation)
- Matplotlib
- NumPy
- SoundFile
- CUDA and NVENC-supporting graphics card (for GPU mode)

### Setup

1. Install required modules:
```bash
pip install torch numpy soundfile librosa matplotlib tqdm omegaconf
```

2. Install FFmpeg (using system package manager):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg
```

3. Ensure the SGVAD model file (`sgvad.pth`) and configuration file (`cfg.yaml`) are in your project directory.

## Basic Usage

### CPU Mode (vad_visualize_cpu.py)

CPU-only mode uses Matplotlib for animation and works in environments without a graphics card.

```bash
python vad_visualize_cpu.py --input "audio_directory/*.wav" --output "output_directory"
```

### GPU Mode (vad_visualize_gpu.py)

GPU-accelerated mode provides fast video encoding using NVENC. Requires an NVIDIA graphics card with CUDA and NVENC support.

```bash
python vad_visualize_gpu.py --input "audio_directory/*.wav" --output "output_directory"
```

## Detailed Parameter Information

### Common Parameters (For Both CPU/GPU Modes)

- `--input`: Input audio file pattern (e.g., "data/*.wav")
- `--output`: Directory to save results
- `--max_files`: (Optional) Maximum number of files to process
- `--seg_duration`: Audio segment length in seconds (default: 30)
- `--merge_threshold`: Threshold for merging short segments in seconds (default: 10)
- `--smoothing_kernel`: Smoothing kernel size (default: 21)
- `--fps`: Video frame rate (default: 30)
- `--workers` or `-w`: Number of processes to use for parallel processing

### GPU Mode Exclusive Parameters

- `--static_plot`: Save visualization results as static images instead of videos

## Comparing CPU and GPU Modes

### CPU Mode (vad_visualize_cpu.py)
- **Advantages**: No graphics card required, compatible with all systems
- **Disadvantages**: Slower processing, especially for video generation (about 100x difference)
- **Use case**: Environments without high-end graphics cards, server environments

### GPU Mode (vad_visualize_gpu.py)
- **Advantages**: Fast video encoding, efficient for batch processing
- **Disadvantages**: Requires NVIDIA graphics card with NVENC support
- **Use case**: Processing large volumes, when high-quality video output is needed
- **Note**: Consumer-grade graphics cards have a default limit on simultaneous encoding sessions, which can be removed using [nvidia-patch](https://github.com/keylase/nvidia-patch)

## Step-by-Step Tutorial

1. **Prepare audio files**:
   - Place WAV files in a directory (16kHz sampling rate recommended)

2. **Create output directory**:
   ```bash
   mkdir -p visualization_results
   ```

3. **Run CPU mode**:
   ```bash
   python vad_visualize_cpu.py --input "your_audio/*.wav" --output "visualization_results" 
   ```

4. **Run GPU mode**:
   ```bash
   python vad_visualize_gpu.py --input "your_audio/*.wav" --output "visualization_results"
   ```

5. **Check results**:
   - Static images: `visualization_results/{filename}/{filename}_plot.png`
   - Videos: `visualization_results/{filename}/{filename}_visualization.mp4`

## Advanced Usage Examples

### Setting smaller segment size and increased merge threshold

```bash
python vad_visualize_gpu.py --input "audio/*.wav" --output "results" --seg_duration 15 --merge_threshold 5
```

### Generate video with higher frame rate

```bash
python vad_visualize_gpu.py --input "audio/*.wav" --output "results" --fps 60
```

### Generate static images only (skip video generation)

```bash
python vad_visualize_gpu.py --input "audio/*.wav" --output "results" --static_plot
```

### Process with more parallel processes (CPU mode)

```bash
python vad_visualize_cpu.py --input "audio/*.wav" --output "results" --workers 16
```

## Visualization Output Description

### Static Plot Configuration
The generated static image consists of two subplots:

1. **Top subplot**: Audio waveform with detected speech segments (red background)
2. **Bottom subplot**: VAD probability scores, 60th and 90th percentile thresholds

### Video Output Features
- Top/bottom subplot configuration is the same as static plot
- Current playback position indicated by a red vertical line
- Synchronized audio playback

## Customization and Tips

### Adjusting Smoothing

The smoothing kernel size affects noise reduction in VAD probability scores:
- Default value of 21 provides moderate smoothing
- Smaller values (e.g., 11) preserve more detailed changes
- Larger values (e.g., 31) provide smoother results

```bash
python vad_visualize_gpu.py --input "audio/*.wav" --output "results" --smoothing_kernel 11
```

### GPU Optimization

When using consumer-grade NVIDIA graphics cards:
- By default, they support only up to 8 simultaneous encoding sessions
- You can apply [nvidia-patch](https://github.com/keylase/nvidia-patch) to utilize more sessions
- When applied, you should allocate more threads through the -w option to maximize performance
- Professional GPUs have no limits or higher limits
- Data center GPUs are not compatible

### Note: Data Center GPUs like A100

Data center GPUs like A100 don't support NVENC, so you must use CPU mode:

```bash
python vad_visualize_cpu.py --input "audio/*.wav" --output "results"
```

## Troubleshooting

### Errors when using GPU mode

1. Verify NVIDIA drivers are up to date (test environment uses driver version 550.120 with CUDA 12.4)
2. Consider using the CPU encoding version
3. Check that NVENC is supported and FFmpeg is working correctly

### Out of memory errors

1. Reduce segment duration: `--seg_duration 15`
2. Limit the number of threads processed at once: `-w 10`

### Timeout when processing long audio files

For very long audio files, adjust segment duration and merge threshold:
```bash
python vad_visualize_gpu.py --input "long_files/*.wav" --output "results" --seg_duration 60 --merge_threshold 15
```

## Output Structure

For each processed audio file (`example.wav`), the output will be:
```
results/
└── example/
    ├── example.wav_00_speech.wav      # First speech segment
    ├── example.wav_01_silence.wav     # First silence segment
    ├── example_plot.png               # Static visualization image
    └── example_visualization.mp4      # Video visualization (if generated)
```

## License

This project is provided for non-commercial use only.
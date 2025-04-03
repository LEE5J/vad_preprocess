# 오디오 세그먼트 프로세서

SGVAD(Stochastic Gates Voice Activity Detection) 모델을 활용하여 오디오 파일에서 발화 부분을 슬라이싱하고 추출하는 도구입니다.
sgvad: arXiv:2210.16022

## 개요

이 프로젝트는 오디오 녹음에서 발화 세그먼트를 자동으로 감지하고 추출하기 위한 강력한 솔루션을 제공합니다. SGVAD 모델을 활용하여 발화 확률 점수를 예측하고, 이를 임계값 기반 알고리즘으로 처리하여 정확하게 발화 세그먼트를 식별합니다.

## 주요 기능

- **정확한 음성 감지**: 강력한 SGVAD 모델을 활용한 신뢰성 높은 음성 활동 감지
- **사용자 정의 임계값**: 퍼센타일 기반 및 절대 임계값 접근 방식 모두 지원
- **시각화**: 오디오 파형과 발화 확률 점수를 보여주는 플롯 생성
- **배치 처리**: 여러 오디오 파일을 한 번에 처리
- **구성 가능한 매개변수**: 조정 가능한 세그먼트 지속 시간, 스무딩 및 병합 임계값

## 프로젝트 설계

프로젝트는 다음 구성 요소로 구성되어 있습니다:

1. **AudioSegmenter**: 오디오를 처리하기 위한 관리 가능한 세그먼트로 분할
2. **VADVisualizer**: 음성 감지 결과의 시각화 생성
3. **VADProcessor**: SGVAD 모델을 사용하여 음성 세그먼트를 식별하는 핵심 프로세서

감지 알고리즘은 다음과 같은 두 단계 접근 방식을 사용합니다:
1. 60번째 퍼센타일 임계값을 적용하여 발화 후보 세그먼트 식별
2. 90번째 퍼센타일 임계값을 적용하여 발화 세그먼트 확인

이 접근 방식은 높은 감지 정확도를 유지하면서 오탐지를 필터링하는 데 도움이 됩니다.

## 설치

### 사전 요구사항

- Python 3.7 이상
- PyTorch
- LibROSA
- SoundFile
- NumPy
- Matplotlib
- tqdm

### 설정

1. 리포지토리 클론:
```bash
git clone https://github.com/LEE5J/vad_preprocess
cd vad_preprocess
```

2. 의존성 설치:
```bash
pip install torch numpy soundfile librosa matplotlib tqdm omegaconf
```

3. SGVAD 모델 파일(`sgvad.pth`)과 구성 파일(`cfg.yaml`)이 프로젝트 디렉토리에 있는지 확인하세요.

## 사용 방법

### 기본 사용법

프로세서를 실행하는 가장 간단한 방법:

```bash
python audio_segment_processor.py --input "오디오_디렉토리/*.wav" --output "출력_디렉토리"
```

### 전체 명령줄 옵션

```bash
python audio_segment_processor.py \
  --input "target_resource/*.wav" \
  --output "preprocess" \
  --max_files 10 \
  --seg_duration 30 \
  --merge_threshold 10 \
  --smoothing_kernel 21
```

### 매개변수 상세 정보

- `--input`: 입력 오디오 파일과 일치하는 패턴(예: "data/*.wav")
- `--output`: 처리된 세그먼트가 저장될 디렉토리
- `--max_files`: (선택 사항) 처리할 최대 파일 수
- `--seg_duration`: 오디오 세그먼트의 지속 시간(초 단위, 기본값: 30)
- `--merge_threshold`: 짧은 세그먼트 병합 임계값(초 단위, 기본값: 10)
- `--smoothing_kernel`: 스무딩 커널 크기(기본값: 21)
- `--no_plot`: 시각화 생성을 비활성화하려면 이 플래그 추가

### 단계별 튜토리얼

1. **오디오 파일 준비**:
   - WAV 오디오 파일을 디렉토리에 배치(예: 

target_resource

)
   - 적절한 품질과 샘플링 레이트(16 kHz 권장)를 갖추었는지 확인

2. **출력 디렉토리 생성**:
   ```bash
   mkdir -p preprocess
   ```

3. **프로세서 실행**:
   ```bash
   python audio_segment_processor.py --input "target_resource/*.wav" --output "preprocess"
   ```

4. **결과 확인**:
   - 각 입력 오디오 파일에는 출력 폴더에 해당 디렉토리가 있습니다.
   - 각 디렉토리 내부에는 다음이 포함됩니다:
     - `filename_XX_speech.wav`로 저장된 음성 세그먼트
     - `filename_XX_silence.wav`로 저장된 무음 세그먼트
     - `filename_plot.png`로 저장된 시각화 플롯(활성화된 경우)

## 사용자 지정 및 팁

### 임계값 선택

기본 구성은 음성 감지에 퍼센타일 기반 임계값을 사용합니다. 그러나 연구 논문에서 언급된 대로 3.5의 고정 임계값이 종종 잘 작동하며 권장됩니다. 이 값은 

cfg.yaml

에서 구성됩니다.

특정 도메인에 따라 임계값 접근 방식을 조정할 수 있습니다:
- 배경 노이즈가 적은 깨끗한 녹음: 낮은 임계값이 잘 작동함
- 노이즈가 많은 환경: 더 높은 임계값이 필요할 수 있음
- 특정 도메인(예: 콜센터 녹음, 팟캐스트): 데이터에 맞게 미세 조정

### 스무딩 구성

기본 스무딩 커널 크기는 21로 설정되어 있으며, 이는 발화 확률 점수의 노이즈를 줄이기 위해 의도적으로 큰 값입니다. 이는 중간 정도의 노이즈가 있는 대부분의 시나리오에서 잘 작동합니다.

그러나:
- 매우 깨끗한 녹음: 발화 세그먼트에서 더 많은 세부 정보를 유지하기 위해 스무딩 커널 크기를 줄이는 것을 고려하세요(예: 11 또는 7)
- 매우 노이지한 녹음: 스무딩을 더 증가시켜야 할 수도 있습니다

스무딩 조정:
```bash
python audio_segment_processor.py --input "오디오_파일/*.wav" --output "결과" --smoothing_kernel 11
```

### 대용량 파일 처리

매우 큰 오디오 파일의 경우, 세그먼트 기능이 효율적인 처리에 도움이 됩니다:
- 기본 세그먼트 지속 시간은 30초입니다
- 병합 임계값(기본값: 10초)보다 짧은 세그먼트는 이전 세그먼트와 병합됩니다

필요에 따라 이러한 매개변수를 조정하세요:
```bash
python audio_segment_processor.py --input "large_files/*.wav" --output "results" --seg_duration 60 --merge_threshold 15
```

## 출력 구조

각 처리된 오디오 파일(`example.wav`)에 대한 출력은 다음과 같습니다:
```
preprocess/
└── example/
    ├── example.wav_00_speech.wav
    ├── example.wav_01_silence.wav
    ├── example.wav_02_speech.wav
    └── example_plot.png
```

## 비디오 시각화 도구

오디오 세그먼트 프로세서 외에도, 이 프로젝트는 오디오와 시각화를 동기화된 비디오로 변환하는 도구를 제공합니다. 이를 통해 분석 결과를 보다 직관적으로 검토할 수 있습니다.

### 주요 기능

- **동기화된 진행 표시**: 오디오 재생에 맞춰 시각화 이미지 위에 진행 선을 표시
- **VAD 시각화**: 오디오 내의 음성 활동 감지 결과를 실시간으로 표시
- **NVIDIA 하드웨어 가속**: 사용 가능한 경우 NVIDIA 하드웨어 인코딩 활용
- **배치 처리**: 여러 오디오-시각화 쌍을 한 번에 처리
- **좌표 로그 지원**: 로그 파일에서 표시 좌표를 자동으로 로드

### 사용 방법

비디오 생성 도구를 실행하는 기본 명령:

```bash
python create_video_with_progress.py --base-dir "/your/project/directory"
```

### 전체 명령줄 옵션

```bash
python create_video_with_progress.py \
  --base-dir "/your/project/directory" \
  --log-file "coordinates.log" \
  --fps 30 \
  --use-vad \
  --vad-threshold 0.5
```

### 매개변수 설명

- `--base-dir`: 시각화 이미지와 오디오 파일이 포함된 기본 디렉토리
- `--log-file`: 표시 좌표가 포함된 로그 파일 경로 (선택사항)
- `--fps`: 출력 비디오의 초당 프레임 수 (기본값: 30)
- `--use-vad`: 시각화에 음성 활동 감지를 표시하려면 이 플래그 추가
- `--vad-threshold`: VAD 활동 감지를 위한 임계값 (기본값: 0.5)

### 디렉토리 구조

비디오 생성 도구는 다음과 같은 디렉토리 구조를 기대합니다:

```
base_directory/
├── target_audio/             # 오디오 파일이 위치하는 디렉토리
│   ├── audio1.wav
│   └── audio2.wav
├── result_compare_plt/       # 시각화 이미지가 위치하는 디렉토리
│   ├── audio1_comparison.png
│   └── audio2_comparison.png
└── result_videos/            # 생성된 비디오가 저장되는 디렉토리 (자동 생성)
    ├── audio1_video.mp4
    └── audio2_video.mp4
```

### 로그 파일 형식

좌표 로그 파일은 다음 형식을 사용해야 합니다:

```
File: audio1, Start: 100, End: 1800, Duration: 5.5s
File: audio2, Start: 120, End: 1750, Duration: 4.2s
```

각 줄에는 파일 이름, 시작 좌표(픽셀), 끝 좌표(픽셀) 및 오디오 지속 시간(초)이 포함되어야 합니다.

## 라이선스

이 프로젝트는 비상업적용 라이선스 하에 제공됩니다.

# Audio Segment Processor

A tool for slicing and extracting speech segments from audio files using the SGVAD (Stochastic Gates Voice Activity Detection) model.

## Overview

This project provides a robust solution for automatically detecting and extracting speech segments from audio recordings. It utilizes the SGVAD model to predict speech probability scores, which are then processed using threshold-based algorithms to identify speech segments accurately.

## Features

- **Accurate Speech Detection**: Leverages the powerful SGVAD model for reliable voice activity detection
- **Customizable Thresholds**: Supports both percentile-based and absolute threshold approaches
- **Visualization**: Generates plots showing audio waveforms and speech probability scores
- **Batch Processing**: Process multiple audio files in one operation
- **Configurable Parameters**: Adjustable segment duration, smoothing, and merge thresholds

## Project Design

The project is structured with the following components:

1. **AudioSegmenter**: Splits audio into manageable segments for processing
2. **VADVisualizer**: Creates visualizations of speech detection results
3. **VADProcessor**: Core processor that identifies speech segments using the SGVAD model

The detection algorithm uses a two-step approach:
1. Apply a 60th percentile threshold to identify candidate speech segments
2. Apply a 90th percentile threshold to confirm speech segments

This approach helps filter out false positives while maintaining high detection accuracy.

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch
- LibROSA
- SoundFile
- NumPy
- Matplotlib
- tqdm

### Setup

1. Clone the repository:
```bash
git clone https://github.com/LEE5J/vad_preprocess
cd vad_preprocess
```

2. Install dependencies:
```bash
pip install torch numpy soundfile librosa matplotlib tqdm omegaconf
```

3. Ensure the SGVAD model file (

sgvad.pth

) and configuration file (

cfg.yaml

) are in the project directory.

## Usage

### Basic Usage

The simplest way to run the processor is:

```bash
python audio_segment_processor.py --input "your_audio_directory/*.wav" --output "output_directory"
```

### Full Command-Line Options

```bash
python audio_segment_processor.py \
  --input "target_resource/*.wav" \
  --output "preprocess" \
  --max_files 10 \
  --seg_duration 30 \
  --merge_threshold 10 \
  --smoothing_kernel 21
```

### Parameter Details

- `--input`: Pattern to match input audio files (e.g., "data/*.wav")
- `--output`: Directory where processed segments will be saved
- `--max_files`: (Optional) Maximum number of files to process
- `--seg_duration`: Duration of audio segments in seconds (default: 30)
- `--merge_threshold`: Threshold for merging short segments in seconds (default: 10)
- `--smoothing_kernel`: Size of smoothing kernel (default: 21)
- `--no_plot`: Add this flag to disable visualization generation

### Step-by-Step Tutorial

1. **Prepare your audio files**:
   - Place your WAV audio files in a directory (e.g., 

target_resource

)
   - Make sure they have a decent quality and sampling rate (16 kHz recommended)

2. **Create an output directory**:
   ```bash
   mkdir -p preprocess
   ```

3. **Run the processor**:
   ```bash
   python audio_segment_processor.py --input "target_resource/*.wav" --output "preprocess"
   ```

4. **Check the results**:
   - Each input audio file will have a corresponding directory in the output folder
   - Inside each directory, you'll find:
     - Speech segments saved as `filename_XX_speech.wav`
     - Silence segments saved as `filename_XX_silence.wav`
     - Visualization plot as `filename_plot.png` (if enabled)

## Customization and Tips

### Threshold Selection

The default configuration uses percentile-based thresholds for speech detection. However, as mentioned in the research paper, a fixed threshold of 3.5 often works well and is recommended. This value is configured in 

cfg.yaml

.

You can adjust the threshold approach based on your specific domain:
- For clean recordings with minimal background noise: Lower thresholds work well
- For noisy environments: Higher thresholds may be necessary
- For specific domains (e.g., call center recordings, podcasts): Fine-tune based on your data

### Smoothing Configuration

The default smoothing kernel size is set to 21, which is intentionally large to reduce noise in the speech probability scores. This works well for most scenarios with moderate noise.

However:
- For very clean recordings: Consider reducing the smoothing kernel size (e.g., 11 or 7) to maintain more detail in the speech segments
- For extremely noisy recordings: You might need to increase the smoothing further

To adjust the smoothing:
```bash
python audio_segment_processor.py --input "your_audio_files/*.wav" --output "results" --smoothing_kernel 11
```

### Processing Large Files

For very large audio files, the segmenting feature helps process them efficiently:
- Default segment duration is 30 seconds
- Segments shorter than the merge threshold (default: 10 seconds) will be merged with the previous segment

Adjust these parameters as needed:
```bash
python audio_segment_processor.py --input "large_files/*.wav" --output "results" --seg_duration 60 --merge_threshold 15
```

## Output Structure

For each processed audio file (`example.wav`), the output will have:
```
preprocess/
└── example/
    ├── example.wav_00_speech.wav
    ├── example.wav_01_silence.wav
    ├── example.wav_02_speech.wav
    └── example_plot.png
```

## Video Visualization Tool

In addition to the audio segment processor, this project provides a tool to convert audio and visualizations into synchronized videos. This allows for a more intuitive review of the analysis results.

### Key Features

- **Synchronized Progress Indicator**: Displays a progress line over the visualization image in sync with audio playback
- **VAD Visualization**: Shows voice activity detection results in real-time within the audio
- **NVIDIA Hardware Acceleration**: Utilizes NVIDIA hardware encoding if available
- **Batch Processing**: Process multiple audio-visualization pairs in one operation
- **Coordinate Log Support**: Automatically loads display coordinates from a log file

### Usage

The basic command to run the video creation tool:

```bash
python create_video_with_progress.py --base-dir "/your/project/directory"
```

### Full Command-Line Options

```bash
python create_video_with_progress.py \
  --base-dir "/your/project/directory" \
  --log-file "coordinates.log" \
  --fps 30 \
  --use-vad \
  --vad-threshold 0.5
```

### Parameter Details

- `--base-dir`: Base directory containing visualization images and audio files
- `--log-file`: Path to the log file containing display coordinates (optional)
- `--fps`: Frames per second for the output video (default: 30)
- `--use-vad`: Add this flag to display voice activity detection in the visualization
- `--vad-threshold`: Threshold for VAD activity detection (default: 0.5)

### Directory Structure

The video creation tool expects the following directory structure:

```
base_directory/
├── target_audio/             # Directory containing audio files
│   ├── audio1.wav
│   └── audio2.wav
├── result_compare_plt/       # Directory containing visualization images
│   ├── audio1_comparison.png
│   └── audio2_comparison.png
└── result_videos/            # Directory where generated videos will be saved (auto-created)
    ├── audio1_video.mp4
    └── audio2_video.mp4
```

### Log File Format

The coordinate log file should use the following format:

```
File: audio1, Start: 100, End: 1800, Duration: 5.5s
File: audio2, Start: 120, End: 1750, Duration: 4.2s
```

Each line should include the file name, start coordinate (pixels), end coordinate (pixels), and audio duration (seconds).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is non commercial License.

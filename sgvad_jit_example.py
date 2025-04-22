#!/usr/bin/env python
import os
import torch
import librosa
from omegaconf import OmegaConf
from nemo.collections.asr.modules import AudioToMFCCPreprocessor

# -----------------------------
# 설정 파일 및 체크포인트 로드
# -----------------------------
config_file = "./cfg.yaml"
if not os.path.exists(config_file):
    raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_file}")

cfg = OmegaConf.load(config_file)

# 장치 결정 (CUDA 사용 가능 시 사용, 아니면 CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg.device = str(device)

if not hasattr(cfg, "ckpt") or not cfg.ckpt:
    raise ValueError(f"{config_file} 파일에 'ckpt' 경로가 설정되어 있지 않습니다.")
if not os.path.exists(cfg.ckpt):
    raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {cfg.ckpt}")

# -----------------------------
# TorchScript VAD 모델 로드
# -----------------------------
script_pt_file = "sgvad_model.pt"
if not os.path.exists(script_pt_file):
    raise FileNotFoundError(f"TorchScript 파일을 찾을 수 없습니다: {script_pt_file}")

print("TorchScript VAD 모델 로드 중...")
model_script = torch.jit.load(script_pt_file, map_location=device)
model_script.eval()

# -----------------------------
# 전처리기 초기화 및 체크포인트에서 상태 로드
# -----------------------------
print("전처리기 초기화 및 상태 로드 중...")
preprocessor = AudioToMFCCPreprocessor(**cfg.preprocessor)
# 체크포인트 로드
ckpt = torch.load(cfg.ckpt, map_location="cpu")
if "preprocessor" not in ckpt:
    raise KeyError(f"체크포인트 '{cfg.ckpt}'에 'preprocessor' state_dict가 없습니다.")
preprocessor.load_state_dict(ckpt["preprocessor"], strict=True)
preprocessor.featurizer.dither = 0.0
preprocessor.featurizer.pad_to = 0
preprocessor.to(device)

# -----------------------------
# 테스트: 오디오 파일 전처리 및 TorchScript 모델 추론
# -----------------------------
# 테스트용 오디오 파일 경로 (경로와 파일명을 필요에 맞게 변경)
test_audio_path = "sample_1.wav"
if not os.path.exists(test_audio_path):
    raise FileNotFoundError(f"테스트 오디오 파일을 찾을 수 없습니다: {test_audio_path}")

print("테스트 오디오 파일 로드 및 전처리 수행 중...")
# librosa를 사용하여 오디오 파일 로드
audio, sr = librosa.load(test_audio_path, sr=cfg.sample_rate)
if audio is None or len(audio) == 0:
    raise ValueError(f"오디오 로드 실패 또는 오디오 길이가 0입니다: {test_audio_path}")

# 오디오 데이터를 텐서로 변환
wave_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
wave_len = torch.tensor([wave_tensor.size(-1)], dtype=torch.long).to(device)

# 전처리기 수행
with torch.no_grad():
    processed_signal, processed_signal_len = preprocessor(input_signal=wave_tensor, length=wave_len)

# TorchScript 모델에 전처리 결과 입력하여 예측 수행
with torch.no_grad():
    output_scores = model_script(processed_signal, processed_signal_len)

print("TorchScript VAD 모델 출력 (프레임별 점수):")
print(output_scores)
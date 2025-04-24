#sgvad_jit.py
import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from nemo.collections.asr.modules import AudioToMFCCPreprocessor, ConvASREncoder
from omegaconf import OmegaConf
import traceback

# -----------------------------
# SGVAD: TorchScript로 변환할 VAD 연산 부분 (전처리된 입력을 받음)
# -----------------------------
class SGVAD(nn.Module):
    """
    전처리된 입력(MFCC 등)을 받아 VAD 연산 및 라벨링 결과를 출력하는 모듈입니다.
    이 모듈은 TorchScript로 추적(tracing)할 대상입니다.
    """
    def __init__(self, model: ConvASREncoder):
        super(SGVAD, self).__init__()
        self.model = model  # 이미 eval()과 freeze()가 이루어진 상태여야 합니다.

    def forward(self, processed_signal: torch.Tensor, processed_signal_len: torch.Tensor) -> torch.Tensor:
        mu, _ = self.model(audio_signal=processed_signal, length=processed_signal_len)
        binary_gates = torch.clamp(mu + 0.5, 0.0, 1.0)
        score = binary_gates.sum(dim=1)
        frame_scores = score / 11.0
        return frame_scores

# -----------------------------
# 설정 및 체크포인트 로드
# -----------------------------
config_file = "./cfg.yaml"
if not os.path.exists(config_file):
    raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_file}")

cfg = OmegaConf.load(config_file)

# 장치 결정 (요청 장치가 CUDA인데 사용 불가하면 CPU로 대체)
requested_device = 'cuda:0'
if requested_device.startswith('cuda') and not torch.cuda.is_available():
    print(f"경고: 요청한 CUDA 장치 '{requested_device}' 사용 불가, CPU로 대체합니다.")
    device = torch.device('cpu')
else:
    device = torch.device(requested_device) if requested_device.startswith('cuda') else torch.device('cpu')
cfg.device = str(device)

# 체크포인트 파일 검증 및 로드
if not hasattr(cfg, 'ckpt') or not cfg.ckpt:
    raise ValueError(f"{config_file} 설정 파일에 'ckpt' 경로가 설정되지 않았습니다.")
if not os.path.exists(cfg.ckpt):
    raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {cfg.ckpt} (설정 파일: {config_file})")

print(f"체크포인트 로드 중: {cfg.ckpt}")
ckpt = torch.load(cfg.ckpt, map_location='cpu')

if 'preprocessor' not in ckpt:
    raise KeyError(f"체크포인트 '{cfg.ckpt}'에 'preprocessor' state_dict가 없습니다.")
if 'vad' not in ckpt:
    raise KeyError(f"체크포인트 '{cfg.ckpt}'에 'vad' state_dict가 없습니다.")

# -----------------------------
# 전처리기 초기화 (AudioToMFCCPreprocessor)
# -----------------------------
print("전처리기 초기화 및 상태 로드 중 (CPU)...")
preprocessor = AudioToMFCCPreprocessor(**cfg.preprocessor)
preprocessor.load_state_dict(ckpt['preprocessor'], strict=True)
preprocessor.featurizer.dither = 0.0
preprocessor.featurizer.pad_to = 0
preprocessor.to(device)

# -----------------------------
# VAD 모델 초기화 (ConvASREncoder)
# -----------------------------
print("VAD 모델 초기화 및 상태 로드 중 (CPU)...")
vad_model = ConvASREncoder(**cfg.vad)
vad_model.load_state_dict(ckpt['vad'], strict=True)
vad_model.eval()
vad_model.freeze()
vad_model.to(device)

# -----------------------------
# 전처리 수행 함수
# -----------------------------
def run_preprocessor(fpath: str):
    """
    주어진 오디오 파일을 로드하고 전처리(MFCC 추출)를 수행합니다.
    """
    try:
        audio, sr = librosa.load(fpath, sr=cfg.sample_rate)
    except Exception as e:
        raise RuntimeError(f"오디오 파일 로드 오류 ({fpath}): {e}")
    if audio is None or len(audio) == 0:
        raise ValueError(f"오디오 로드 실패 또는 길이가 0입니다: {fpath}")
    wave_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    wave_len = torch.tensor([wave_tensor.size(-1)], dtype=torch.long)
    wave_tensor = wave_tensor.to(device)
    wave_len = wave_len.to(device)
    processed_signal, processed_signal_len = preprocessor(input_signal=wave_tensor, length=wave_len)
    return processed_signal, processed_signal_len

# -----------------------------
# 전처리 및 TorchScript 추적 수행
# -----------------------------
# 예제용 오디오 파일 경로 (적절히 변경)
dummy_audio_path = "sample_1.wav"
if not os.path.exists(dummy_audio_path):
    raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {dummy_audio_path}")

print("전처리 수행 중...")
processed_signal, processed_signal_len = run_preprocessor(dummy_audio_path)

# SGVAD 인스턴스 생성 (전처리된 입력을 받는 연산부)
sgvad_model = SGVAD(vad_model).to(device)
sgvad_model.eval()

print("전처리된 결과부터 시작하는 VAD 연산 부분을 TorchScript로 추적(tracing)합니다...")
try:
    scripted_sgvad = torch.jit.trace(sgvad_model, (processed_signal, processed_signal_len))
    script_pt_file = "sgvad_model.pt"
    scripted_sgvad.save(script_pt_file)
    print(f"TorchScript 모델이 성공적으로 저장되었습니다: {script_pt_file}")
except Exception as e:
    print(f"TorchScript 변환 및 저장 중 오류 발생: {e}")
    traceback.print_exc()

# 예시: 저장된 TorchScript 모델로 예측 수행
with torch.no_grad():
    output_scores = scripted_sgvad(processed_signal, processed_signal_len)
    print("TorchScript 모델 출력:", output_scores)
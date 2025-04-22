import os
import torch
import librosa
from omegaconf import OmegaConf
from nemo.collections.asr.modules import AudioToMFCCPreprocessor, ConvASREncoder

# -----------------------------
# 설정 및 체크포인트 로드
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

print("체크포인트 로드 중...")
ckpt = torch.load(cfg.ckpt, map_location="cpu")

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
# 전처리기 초기화 및 상태 로드
# -----------------------------
print("전처리기 초기화 및 상태 로드 중...")
preprocessor = AudioToMFCCPreprocessor(**cfg.preprocessor)
if "preprocessor" not in ckpt:
    raise KeyError(f"체크포인트 '{cfg.ckpt}'에 'preprocessor' state_dict가 없습니다.")
preprocessor.load_state_dict(ckpt["preprocessor"], strict=True)
preprocessor.featurizer.dither = 0.0
preprocessor.featurizer.pad_to = 0
preprocessor.to(device)

# -----------------------------
# 원본 VAD 모델 (ConvASREncoder) 초기화 및 상태 로드
# -----------------------------
print("원본 VAD 모델 초기화 및 상태 로드 중...")
vad_model = ConvASREncoder(**cfg.vad)
if "vad" not in ckpt:
    raise KeyError(f"체크포인트 '{cfg.ckpt}'에 'vad' state_dict가 없습니다.")
vad_model.load_state_dict(ckpt["vad"], strict=True)
vad_model.eval()
vad_model.freeze()
vad_model.to(device)

# -----------------------------
# 테스트용 오디오 파일 전처리 함수
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
        raise ValueError(f"오디오 로드 실패 또는 오디오 길이가 0입니다: {fpath}")
    wave_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    wave_len = torch.tensor([wave_tensor.size(-1)], dtype=torch.long).to(device)
    processed_signal, processed_signal_len = preprocessor(input_signal=wave_tensor, length=wave_len)
    return processed_signal, processed_signal_len

# -----------------------------
# 테스트 수행: 전처리 및 두 모델 예측
# -----------------------------
# 테스트용 오디오 파일 경로 (필요에 맞게 변경)
test_audio_path = "sample_1.wav"
if not os.path.exists(test_audio_path):
    raise FileNotFoundError(f"테스트 오디오 파일을 찾을 수 없습니다: {test_audio_path}")

print("테스트 오디오 파일 로드 및 전처리 수행 중...")
processed_signal, processed_signal_len = run_preprocessor(test_audio_path)

# TorchScript VAD 모델 예측
with torch.no_grad():
    output_script = model_script(processed_signal, processed_signal_len)

# 원본 VAD 모델 예측
with torch.no_grad():
    # 원본 모델은 전처리된 입력을 그대로 사용할 수 있음
    mu, _ = vad_model(audio_signal=processed_signal, length=processed_signal_len)
    binary_gates = torch.clamp(mu + 0.5, 0.0, 1.0)
    score = binary_gates.sum(dim=1)
    output_orig = score / 11.0

print("\n--- 예측 결과 비교 ---")
print("TorchScript 모델 출력 (프레임별 점수):")
print(output_script)
print("\n원본 VAD 모델 출력 (프레임별 점수):")
print(output_orig)

# 결과 비교 (숫자 차이가 거의 없으면 True)
are_close = torch.allclose(output_script, output_orig, atol=1e-4)
print(f"\n두 결과가 동일한가요? {are_close}")
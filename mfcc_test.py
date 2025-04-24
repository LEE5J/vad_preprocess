import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from nemo.collections.asr.modules import AudioToMFCCPreprocessor
from omegaconf import OmegaConf

def compare_preprocessor_versions():
    """
    체크포인트로 로드한 AudioToMFCCPreprocessor와 
    기본 초기화된 AudioToMFCCPreprocessor를 비교합니다.
    """
    # 파일 경로 및 설정
    audio_path = "sample_1.wav"
    config_file = "./cfg.yaml"
    
    # 파일 존재 확인
    if not os.path.exists(audio_path) or not os.path.exists(config_file):
        raise FileNotFoundError(f"필요한 파일이 없습니다.")
    
    # 설정 로드
    cfg = OmegaConf.load(config_file)
    
    # MFCC 설정
    sample_rate = 16000
    window_size = 0.025
    window_stride = 0.01
    window = "hann"
    n_fft = 512
    n_mels = 32
    n_mfcc = n_mels
    
    preprocessor_config = {
        "sample_rate": sample_rate,
        "window_size": window_size,
        "window_stride": window_stride,
        "window": window,
        "n_mels": n_mels,
        "n_mfcc": n_mfcc,
        "n_fft": n_fft,
    }
    
    # 오디오 로드
    print(f"오디오 파일 로드 중: {audio_path} (샘플링 레이트: {sample_rate}Hz)")
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if audio is None or len(audio) == 0:
        raise ValueError(f"오디오 로드 실패 또는 길이가 0입니다: {audio_path}")
    
    wave_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    wave_len = torch.tensor([wave_tensor.size(-1)], dtype=torch.long)
    
    # 1. 먼저 기본 전처리기로 MFCC 계산
    print("\n1단계: 기본 전처리기로 MFCC 계산 중...")
    base_preprocessor = AudioToMFCCPreprocessor(**preprocessor_config)
    base_preprocessor.eval()
    
    with torch.no_grad():
        base_mfcc, base_mfcc_len = base_preprocessor(input_signal=wave_tensor, length=wave_len)
        base_mfcc_np = base_mfcc.squeeze(0).numpy()
    
    # 기본 전처리기의 주요 버퍼 저장
    base_mel_basis = base_preprocessor.featurizer.mel_basis.detach().cpu().numpy() if hasattr(base_preprocessor.featurizer, 'mel_basis') else None
    base_dct_matrix = base_preprocessor.featurizer.dct_matrix.detach().cpu().numpy() if hasattr(base_preprocessor.featurizer, 'dct_matrix') else None
    base_window = base_preprocessor.featurizer.window.detach().cpu().numpy() if hasattr(base_preprocessor.featurizer, 'window') else None
    
    # 메모리에서 기본 전처리기 제거
    del base_preprocessor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 2. 체크포인트 로드 후 전처리기로 MFCC 계산
    print("\n2단계: 체크포인트 로드 후 MFCC 계산 중...")
    
    # 체크포인트 로드
    if not hasattr(cfg, 'ckpt') or not cfg.ckpt:
        raise ValueError(f"설정 파일에 'ckpt' 경로가 설정되지 않았습니다.")
    if not os.path.exists(cfg.ckpt):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {cfg.ckpt}")
    
    print(f"체크포인트 로드 중: {cfg.ckpt}")
    ckpt = torch.load(cfg.ckpt, map_location='cpu')
    
    if 'preprocessor' not in ckpt:
        raise KeyError(f"체크포인트에 'preprocessor' state_dict가 없습니다.")
    
    # 로드된 전처리기 초기화
    loaded_preprocessor = AudioToMFCCPreprocessor(**preprocessor_config)
    loaded_preprocessor.load_state_dict(ckpt['preprocessor'], strict=True)
    loaded_preprocessor.eval()
    
    # 로드된 전처리기로 MFCC 계산
    with torch.no_grad():
        loaded_mfcc, loaded_mfcc_len = loaded_preprocessor(input_signal=wave_tensor, length=wave_len)
        loaded_mfcc_np = loaded_mfcc.squeeze(0).numpy()
    
    # 로드된 전처리기의 주요 버퍼 저장
    loaded_mel_basis = loaded_preprocessor.featurizer.mel_basis.detach().cpu().numpy() if hasattr(loaded_preprocessor.featurizer, 'mel_basis') else None
    loaded_dct_matrix = loaded_preprocessor.featurizer.dct_matrix.detach().cpu().numpy() if hasattr(loaded_preprocessor.featurizer, 'dct_matrix') else None
    loaded_window = loaded_preprocessor.featurizer.window.detach().cpu().numpy() if hasattr(loaded_preprocessor.featurizer, 'window') else None
    
    # 3. 결과 비교
    print("\n3단계: 두 전처리기의 MFCC 결과 비교 중...")
    print(f"기본 MFCC 형태: {base_mfcc_np.shape}")
    print(f"로드된 MFCC 형태: {loaded_mfcc_np.shape}")
    
    min_time = min(base_mfcc_np.shape[1], loaded_mfcc_np.shape[1])
    base_mfcc_trimmed = base_mfcc_np[:, :min_time]
    loaded_mfcc_trimmed = loaded_mfcc_np[:, :min_time]
    
    abs_diff = np.abs(base_mfcc_trimmed - loaded_mfcc_trimmed)
    mean_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    std_diff = np.std(abs_diff)
    
    print(f"\nMFCC 차이 통계:")
    print(f"  평균 절대 차이: {mean_diff:.6f}")
    print(f"  최대 절대 차이: {max_diff:.6f}")
    print(f"  차이의 표준편차: {std_diff:.6f}")
    
    # 상관관계 분석
    correlation = np.corrcoef(base_mfcc_trimmed.flatten(), loaded_mfcc_trimmed.flatten())[0, 1]
    print(f"  두 MFCC 간 상관계수: {correlation:.6f}")
    
    # 4. 주요 내부 파라미터 비교 (mel_basis, dct_matrix, window)
    print("\n4단계: 주요 내부 파라미터 비교 중...")
    
    # 파라미터 비교 및 시각화 함수
    def compare_param(name, base_param, loaded_param):
        if base_param is None or loaded_param is None:
            print(f"  {name} 비교 불가: 파라미터가 존재하지 않습니다.")
            return
        
        abs_diff = np.abs(base_param - loaded_param)
        mean_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
        
        print(f"\n{name} 비교:")
        print(f"  형태: {loaded_param.shape}")
        print(f"  평균 절대 차이: {mean_diff:.6f}")
        print(f"  최대 절대 차이: {max_diff:.6f}")
        
        # 시각화
        plt.figure(figsize=(15, 10))
        
        # 1D 또는 2D 데이터에 따라 시각화 방법 선택
        if len(loaded_param.shape) == 1 or (len(loaded_param.shape) == 2 and (loaded_param.shape[0] == 1 or loaded_param.shape[1] == 1)):
            # 1D 데이터 또는 1행/1열의 2D 데이터
            data_1d = loaded_param.flatten()
            base_1d = base_param.flatten()
            
            plt.subplot(311)
            plt.title(f"기본 {name}")
            plt.plot(base_1d)
            
            plt.subplot(312)
            plt.title(f"로드된 {name}")
            plt.plot(data_1d)
            
            plt.subplot(313)
            plt.title("절대 차이")
            plt.plot(np.abs(data_1d - base_1d))
        else:
            # 2D 데이터
            plt.subplot(311)
            plt.title(f"기본 {name}")
            plt.imshow(base_param, aspect='auto', origin='lower')
            plt.colorbar()
            
            plt.subplot(312)
            plt.title(f"로드된 {name}")
            plt.imshow(loaded_param, aspect='auto', origin='lower')
            plt.colorbar()
            
            plt.subplot(313)
            plt.title("절대 차이")
            plt.imshow(abs_diff, aspect='auto', origin='lower')
            plt.colorbar()
        
        plt.tight_layout()
        filename = f"{name.replace(' ', '_').lower()}_comparison.png"
        plt.savefig(filename)
        plt.close()
        print(f"  {name} 비교가 '{filename}'에 저장되었습니다.")
        
        return mean_diff, max_diff
    
    # 주요 파라미터 비교
    mel_basis_stats = compare_param("Mel 필터뱅크", base_mel_basis, loaded_mel_basis)
    dct_matrix_stats = compare_param("DCT 행렬", base_dct_matrix, loaded_dct_matrix)
    window_stats = compare_param("Window", base_window, loaded_window)
    
    # 5. MFCC 시각화
    plt.figure(figsize=(15, 10))
    
    plt.subplot(311)
    plt.title("기본 전처리기 MFCC")
    plt.imshow(base_mfcc_trimmed, aspect='auto', origin='lower')
    plt.colorbar()
    
    plt.subplot(312)
    plt.title("로드된 전처리기 MFCC")
    plt.imshow(loaded_mfcc_trimmed, aspect='auto', origin='lower')
    plt.colorbar()
    
    plt.subplot(313)
    plt.title("MFCC 절대 차이")
    plt.imshow(abs_diff, aspect='auto', origin='lower')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("mfcc_comparison.png")
    plt.close()
    print(f"MFCC 비교가 'mfcc_comparison.png'에 저장되었습니다.")
    
    # 6. 각 MFCC 계수별 차이 분석
    print("\n각 MFCC 계수별 차이:")
    coef_diffs = []
    for i in range(n_mfcc):
        coef_mean_diff = np.mean(np.abs(base_mfcc_trimmed[i, :] - loaded_mfcc_trimmed[i, :]))
        coef_max_diff = np.max(np.abs(base_mfcc_trimmed[i, :] - loaded_mfcc_trimmed[i, :]))
        coef_diffs.append((i, coef_mean_diff, coef_max_diff))
        print(f"  MFCC 계수 {i}의 평균 절대 차이: {coef_mean_diff:.6f}, 최대 차이: {coef_max_diff:.6f}")
    
    # 차이가 가장 큰 계수들 식별
    sorted_diffs = sorted(coef_diffs, key=lambda x: x[1], reverse=True)
    print("\n차이가 가장 큰 MFCC 계수 (평균 차이 기준):")
    for i, mean_diff, max_diff in sorted_diffs[:5]:  # 상위 5개
        print(f"  MFCC 계수 {i}: 평균 차이 {mean_diff:.6f}, 최대 차이 {max_diff:.6f}")
    
    # 7. 특정 프레임 자세히 비교
    if min_time > 0:
        frame_indices = [0, min_time // 2, min_time - 1]  # 첫 번째, 중간, 마지막 프레임
        
        for idx in frame_indices:
            print(f"\n프레임 {idx} 비교:")
            print("기본 MFCC:")
            print(base_mfcc_trimmed[:, idx])
            print("로드된 MFCC:")
            print(loaded_mfcc_trimmed[:, idx])
            print("절대 차이:")
            frame_diff = np.abs(base_mfcc_trimmed[:, idx] - loaded_mfcc_trimmed[:, idx])
            print(frame_diff)
            print(f"이 프레임의 평균 차이: {np.mean(frame_diff):.6f}, 최대 차이: {np.max(frame_diff):.6f}")
    
    # 8. 종합 보고서 작성
    with open("preprocessor_comparison_report.txt", "w") as f:
        f.write("AudioToMFCCPreprocessor 비교 분석 보고서\n")
        f.write("=====================================\n\n")
        
        f.write("1. MFCC 결과 차이 통계:\n")
        f.write(f"  평균 절대 차이: {mean_diff:.6f}\n")
        f.write(f"  최대 절대 차이: {max_diff:.6f}\n")
        f.write(f"  차이의 표준편차: {std_diff:.6f}\n")
        f.write(f"  상관계수: {correlation:.6f}\n")
        
        f.write("\n2. 주요 파라미터 차이:\n")
        if mel_basis_stats:
            f.write(f"  Mel 필터뱅크: 평균 차이 {mel_basis_stats[0]:.6f}, 최대 차이 {mel_basis_stats[1]:.6f}\n")
        if dct_matrix_stats:
            f.write(f"  DCT 행렬: 평균 차이 {dct_matrix_stats[0]:.6f}, 최대 차이 {dct_matrix_stats[1]:.6f}\n")
        if window_stats:
            f.write(f"  Window: 평균 차이 {window_stats[0]:.6f}, 최대 차이 {window_stats[1]:.6f}\n")
        
        f.write("\n3. 각 MFCC 계수별 차이:\n")
        for i, mean_diff, max_diff in coef_diffs:
            f.write(f"  MFCC 계수 {i}의 평균 절대 차이: {mean_diff:.6f}, 최대 차이: {max_diff:.6f}\n")
        
        f.write("\n4. 차이가 가장 큰 MFCC 계수 (평균 차이 기준):\n")
        for i, mean_diff, max_diff in sorted_diffs[:5]:  # 상위 5개
            f.write(f"  MFCC 계수 {i}: 평균 차이 {mean_diff:.6f}, 최대 차이 {max_diff:.6f}\n")
        
        f.write("\n5. 프레임별 분석:\n")
        for idx in frame_indices:
            frame_diff = np.abs(base_mfcc_trimmed[:, idx] - loaded_mfcc_trimmed[:, idx])
            f.write(f"\n프레임 {idx}:\n")
            f.write(f"  평균 차이: {np.mean(frame_diff):.6f}, 최대 차이: {np.max(frame_diff):.6f}\n")
            f.write("  기본 MFCC 값:\n")
            f.write(f"  {base_mfcc_trimmed[:, idx].tolist()}\n")
            f.write("  로드된 MFCC 값:\n")
            f.write(f"  {loaded_mfcc_trimmed[:, idx].tolist()}\n")
            f.write("  절대 차이:\n")
            f.write(f"  {frame_diff.tolist()}\n")
    
    print("\n자세한 분석이 'preprocessor_comparison_report.txt' 파일에 저장되었습니다.")
    
    return base_mfcc_np, loaded_mfcc_np

if __name__ == "__main__":
    compare_preprocessor_versions()
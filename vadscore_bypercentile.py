#vadscore_bypercentile.py
# 퍼센타일 기반의 snr 비율에 대한 vad의 점수를 생성한다
from glob import glob
import os
import librosa
from nemo.collections.asr.modules import AudioToMFCCPreprocessor
import random
import numpy as np
import torch
from tqdm import tqdm
from parse_textgrid import parse_textgrid_to_labels
from sgvad import SGVAD
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
import pandas as pd

device = "cuda"
LABEL_TG_DIR = "./label_tg" 
NOISE_MIXED_AUDIO_DIR = "./noise_mixed_audio2" 
RANDOM_SEED = 42
sample_rate = 16000
MAX_FILES_PER_FOLDER = float('inf')
CSV_FILENAME = "snr_accuracy_data.csv"

def get_mfcc(wave_tensor,wave_len,sample_rate=16000):
    # 오디오 fp32 tensor 로 입력 (-1) -> (32,-1) MFCC 텐서 반환
    preprocessor_config = {
        "sample_rate": sample_rate,
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "n_mels": 32,
        "n_mfcc": 32,
        "n_fft": 512,
    }
    preprocessor = AudioToMFCCPreprocessor(**preprocessor_config).to(device)
    mfcc, mfcc_len = preprocessor(input_signal=wave_tensor, length=wave_len)
    return mfcc.to(device), mfcc_len.to(device)

def get_score(force_recalculate=False):
    """
    SNR 비율과 정확도 점수를 계산하고 CSV 파일로 저장
    
    Args:
        force_recalculate: 기존 CSV 파일이 있어도 다시 계산할지 여부
    
    Returns:
        tuple: (snr_values, accuracy_values) 튜플
    """
    # 이미 계산된 데이터가 있으면 로드
    if os.path.exists(CSV_FILENAME) and not force_recalculate:
        print(f"기존 계산된 데이터 '{CSV_FILENAME}'을 로드합니다.")
        df = pd.read_csv(CSV_FILENAME)
        return df['snr'].tolist(), df['accuracy'].tolist()
    
    # 1. 랜덤 시드 결정
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 오디오 경로 찾기
    all_audio_files = []
    subfolder_files = {} # 폴더명 -> 파일 경로 리스트 맵
    try:
        for subfolder_path in glob(os.path.join(NOISE_MIXED_AUDIO_DIR, '*/')):
            if not os.path.isdir(subfolder_path): continue # 디렉토리가 아니면 건너띔
            subfolder_name = os.path.basename(os.path.normpath(subfolder_path))
            wav_files = glob(os.path.join(subfolder_path, '*.wav'))
            if not wav_files:
                continue
            # 파일 샘플링
            num_files_to_sample = min(len(wav_files), MAX_FILES_PER_FOLDER)
            sampled_files = random.sample(wav_files, num_files_to_sample)
            subfolder_files[subfolder_name] = sampled_files
            all_audio_files.extend(sampled_files)
    except Exception as e:
        print(f"오류: 오디오 파일 검색 중 오류 발생: {e}")
        return [], []
    if not all_audio_files:
        print("오류: 테스트할 오디오 파일을 찾을 수 없습니다.")
        return [], []
    
    print(f"테스트할 오디오 파일 수: {len(all_audio_files)}")
    
    # 2. 모델 초기화
    sgvad = SGVAD()
    all_labels = sgvad.predict(all_audio_files)
    snr_score_pairs = []
    
    for audio_path, pred_label in tqdm(zip(all_audio_files, all_labels)):
        if "test" in audio_path:
            continue # test 들어간 파일은 textgrid 없음
        
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        audio_abs = np.abs(audio)
        percentile_15 = np.percentile(audio_abs, 15)
        percentile_95 = np.percentile(audio_abs, 95)
        
        # 비율 계산 (95 퍼센타일 / 15 퍼센타일)
        # 0으로 나누는 것을 방지
        if percentile_15 > 0:
            snr = percentile_95 / percentile_15
        else:
            snr = float('inf')
            
        tg_label, _ = parse_textgrid_to_labels(os.path.join(LABEL_TG_DIR, os.path.basename(audio_path).replace(".wav", ".TextGrid")))
        if tg_label is None:
            print(f"오류: 레이블을 가져올 수 없습니다: {audio_path}")
            continue
            
        min_len = min(len(tg_label), len(pred_label))
        accuracy = accuracy_score(tg_label[:min_len], pred_label[:min_len])
        snr_score_pairs.append((snr, accuracy))
    
    # CSV 파일로 저장
    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['snr', 'accuracy'])
        writer.writerows(snr_score_pairs)
    
    print(f"SNR과 정확도 데이터를 '{CSV_FILENAME}'에 저장했습니다.")
    
    # snr과 accuracy 값 추출
    snr_values = [pair[0] for pair in snr_score_pairs]
    accuracy_values = [pair[1] for pair in snr_score_pairs]
    
    return snr_values, accuracy_values

def visualize(snr_values, accuracy_values, output_filename='snr_accuracy.png'):
    """
    SNR 값과 정확도 데이터를 시각화하여 이미지 파일로 저장
    
    Args:
        snr_values: SNR 값 리스트
        accuracy_values: 정확도 값 리스트
        output_filename: 출력 이미지 파일 이름
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(snr_values, accuracy_values, alpha=0.1, s=1)
    
    plt.title('SNR vs Accuracy')
    plt.xlabel('SNR(95p/15p)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # y축 범위 0~1로 고정
    plt.ylim(0, 1)
    
    # x축 범위 0~80으로 고정
    plt.xlim(0, 80)
    
    # x축 로그 스케일 적용 (SNR 값 범위가 넓을 경우)
    if max(snr_values) / min(snr_values) > 100:
        plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"그래프를 '{output_filename}'로 저장했습니다.")

def main():
    # 데이터 계산 및 저장 (또는 기존 데이터 로드)
    snr_values, accuracy_values = get_score(force_recalculate=False)
    
    # 시각화
    if snr_values and accuracy_values:
        visualize(snr_values, accuracy_values)
    else:
        print("시각화할 데이터가 없습니다.")

if __name__ == "__main__":
    main()
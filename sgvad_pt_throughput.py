import os
import time
import glob
from nemo.collections.asr.modules import AudioToMFCCPreprocessor
import numpy as np
import torch
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_segmented_audio(audio, label, smooth=15):
    # Modified to skip saving files
    threshold_low = 0.15
    threshold_high = 0.2
    raw_label = label.squeeze(0).cpu().numpy()
    
    kernel = np.ones(smooth) / smooth
    if len(raw_label) < smooth:
        label = raw_label
    else:
        label = np.convolve(raw_label, kernel, mode='same')

    candidate_labels = (label > threshold_low).astype(int)
    final_labels = np.zeros_like(candidate_labels)
    
    i = 0
    while i < len(candidate_labels):
        if candidate_labels[i] == 1:
            start_frame_idx = i
            while i < len(candidate_labels) and candidate_labels[i] == 1:
                i += 1
            end_frame_idx = i - 1
            candidate_segment = label[start_frame_idx:end_frame_idx+1]
            if np.any(candidate_segment >= threshold_high):
                final_labels[start_frame_idx:end_frame_idx+1] = 1
        else:
            i += 1

    current_label = final_labels[0]
    start_frame_idx = 0
    segments_info = []
    
    for i, label_val in enumerate(final_labels):
        if label_val == current_label:
            continue
        else:
            segments_info.append((current_label, start_frame_idx, i - 1))
            start_frame_idx = i
            current_label = label_val
    segments_info.append((current_label, start_frame_idx, len(final_labels) - 1))
    
    return segments_info  # Just return segments, don't save files

def run_mfcc_benchmark(wave_tensor, wave_len, num_runs=2000):
    sample_rate = 16000
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
    
    # num_runs 만큼 반복 처리
    for _ in range(num_runs):
        mfcc, mfcc_len = preprocessor(input_signal=wave_tensor, length=wave_len)
    
    # 마지막 결과만 반환
    return mfcc, mfcc_len

def run_inference_benchmark(mfcc, mfcc_len, num_runs=2000):
    # 함수 내에서 초기화
    script_pt_file = "sgvad_model.pt"
    model = torch.jit.load(script_pt_file, map_location=device)
    model.eval()
    # num_runs 만큼 반복 처리
    for _ in range(num_runs):
        label = model(mfcc, mfcc_len)
    # 마지막 결과만 반환
    return label

def run_segmentation_benchmark(audio, label, num_runs=2000):
    smooth =15
    threshold_low = 0.15
    threshold_high = 0.2
    raw_label = label.squeeze(0).cpu().numpy()
    # num_runs 만큼 반복 처리
    for _ in range(num_runs):
        kernel = np.ones(smooth) / smooth
        if len(raw_label) < smooth:
            label = raw_label
        else:
            label = np.convolve(raw_label, kernel, mode='same')

        candidate_labels = (label > threshold_low).astype(int)
        final_labels = np.zeros_like(candidate_labels)
        
        i = 0
        while i < len(candidate_labels):
            if candidate_labels[i] == 1:
                start_frame_idx = i
                while i < len(candidate_labels) and candidate_labels[i] == 1:
                    i += 1
                end_frame_idx = i - 1
                candidate_segment = label[start_frame_idx:end_frame_idx+1]
                if np.any(candidate_segment >= threshold_high):
                    final_labels[start_frame_idx:end_frame_idx+1] = 1
            else:
                i += 1

        current_label = final_labels[0]
        start_frame_idx = 0
        segments_info = []
        
        for i, label_val in enumerate(final_labels):
            if label_val == current_label:
                continue
            else:
                segments_info.append((current_label, start_frame_idx, i - 1))
                start_frame_idx = i
                current_label = label_val
            segments_info.append((current_label, start_frame_idx, len(final_labels) - 1))
    # 마지막 결과만 반환
    return segments_info

def find_files_by_duration(directory, target_durations, tolerance=0.1):
    """Find files with durations close to target values"""
    result = {}
    wav_files = glob.glob(os.path.join(directory, "*.wav"))
    
    for wav_file in wav_files:
        audio, sr = librosa.load(wav_file, sr=None)
        duration = len(audio) / sr
        
        for target in target_durations:
            if target not in result and abs(duration - target) <= tolerance:
                result[target] = (wav_file, duration)
                print(f"Found file for {target}s: {os.path.basename(wav_file)} ({duration:.3f}s)")
                break
    
    return result

def benchmark_pipeline(audio_path, actual_duration, num_runs=20000):
    sample_rate = 16000
    
    # 오디오 로딩 및 텐서 변환 (시간 측정 안 함)
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    wave_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    wave_len = torch.tensor([wave_tensor.size(-1)], dtype=torch.long).to(device)
    
    # MFCC 추출 벤치마크 (외부에서 시간 측정)
    start_time = time.time()
    mfcc, mfcc_len = run_mfcc_benchmark(wave_tensor, wave_len, num_runs)
    mfcc_total_time = time.time() - start_time
    
    # 모델 추론 벤치마크 (외부에서 시간 측정)
    start_time = time.time()
    label = run_inference_benchmark(mfcc, mfcc_len, num_runs)
    inference_total_time = time.time() - start_time
    
    # 세그멘테이션 벤치마크 (외부에서 시간 측정)
    start_time = time.time()
    segments = run_segmentation_benchmark(audio, label, num_runs)
    segmentation_total_time = time.time() - start_time
    
    # 결과 출력
    total_pipeline_time = mfcc_total_time + inference_total_time + segmentation_total_time
    
    print(f"\n--- Results for {os.path.basename(audio_path)} (duration: {actual_duration:.3f}s) ---")
    print(f"MFCC extraction: total={mfcc_total_time:.6f}s for {num_runs} runs ({mfcc_total_time/num_runs:.6f}s per run)")
    print(f"Model inference: total={inference_total_time:.6f}s for {num_runs} runs ({inference_total_time/num_runs:.6f}s per run)")
    print(f"Segmentation: total={segmentation_total_time:.6f}s for {num_runs} runs ({segmentation_total_time/num_runs:.6f}s per run)")
    print(f"Total pipeline time: {total_pipeline_time:.6f}s for {num_runs} runs")
    print(f"Average time per complete pipeline: {num_runs/total_pipeline_time:.6f}s")
    
    # 처리량 계산
    throughput = num_runs / total_pipeline_time
    print(f"Throughput: {throughput:.2f} inferences/second")
    print(f"Processing time ratio: {total_pipeline_time/(num_runs*actual_duration):.4f}x realtime")

def main():
    # Directory containing WAV files
    audio_dir = "sample_noise_mixed_audio/normal_noise_1"
    target_durations = [2, 3, 4, 5, 6]  # Target durations in seconds
    
    # Find files with appropriate durations
    files_by_duration = find_files_by_duration(audio_dir, target_durations)
    
    # Run benchmark for each file
    print(f"\nRunning benchmarks with {device} device...")
    for duration, (file_path, actual_duration) in files_by_duration.items():
        benchmark_pipeline(file_path, actual_duration)

if __name__ == "__main__":
    main()
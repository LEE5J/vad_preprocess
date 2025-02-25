import os
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sgvad import SGVAD
import matplotlib.pyplot as plt


def split_audio_segments(audio, sample_rate, seg_duration=30, merge_threshold=10):
    seg_samples = int(sample_rate * seg_duration)
    merge_samples = int(sample_rate * merge_threshold)
    total_samples = len(audio)
    if total_samples < seg_samples:
        return [audio]
    segments = []
    start = 0
    while start < total_samples:
        end = start + seg_samples
        if end >= total_samples:
            # 마지막 세그먼트의 길이가 merge_threshold 미만인 경우, 이전 세그먼트와 병합
            if (total_samples - start) < merge_samples and segments:
                prev_segment = segments.pop()
                # 이전 세그먼트와 현재 나머지 오디오를 이어 붙여서 하나의 세그먼트로 만듦
                segments.append(np.concatenate((prev_segment, audio[start:total_samples])))
            else:
                segments.append(audio[start:total_samples])
            break
        else:
            segments.append(audio[start:end])
            start = end
    return segments



def process_audio_file(fpath, sgvad, out_dir, seg_duration=30, merge_threshold=10, smoothing_kernel=21):
    # 오디오 로드 (librosa 기반)
    audio = sgvad.load_audio(fpath)
    audio = np.array(audio)
    filename = os.path.basename(fpath)
    sample_rate = sgvad.cfg.sample_rate
    segments = split_audio_segments(audio, sample_rate, seg_duration, merge_threshold) # 오디오 원문 제공
    base_filename = os.path.splitext(os.path.basename(fpath))[0]
    
    # VAD 모델의 프레임 레이트 계산 (10ms 프레임 가정)
    frame_rate_ms = 10  # 10ms per frame
    samples_per_frame = int(frame_rate_ms / 1000 * sample_rate)  # 16kHz에서는 160 samples
    
    candidate_labels = []  # 60퍼센타일 기준 발화 후보군 라벨
    all_probs = []  # 모든 세그먼트의 확률값을 저장할 리스트
    segment_boundaries = []  # 세그먼트 경계 저장
    segment_thresholds_60 = []  # 각 세그먼트의 60퍼센타일 임계값 저장
    
    current_position = 0
    for segment in segments:
        # 세그먼트 단위로 VAD 확률 계산
        probs = sgvad.predict(segment, smoothing_kernel)  # (frame_num, 2)
        probs = np.array(probs).ravel()
        
        # 세그먼트 정보 저장
        segment_length = len(probs)
        segment_boundaries.append((current_position, current_position + segment_length))
        threshold_60 = np.percentile(probs[:], 60)
        segment_thresholds_60.append(threshold_60)
        current_position += segment_length
        
        all_probs.extend(probs.tolist())  # 확률값 저장
        # 60퍼센타일 기준으로 발화 후보군 라벨 생성
        binary_label = (probs[:] > threshold_60).astype(int)
        candidate_labels.extend(binary_label.tolist())
    
    candidate_labels = np.array(candidate_labels)
    all_probs = np.array(all_probs)  # 모든 확률값
    
    # 전체 데이터의 90 퍼센타일 임계값 계산
    global_threshold_90 = np.percentile(all_probs, 90)
    
    # 최종 라벨 초기화 (일단 모두 0으로)
    final_labels = np.zeros_like(candidate_labels)
    
    # 발화 후보군 식별 및 검증
    i = 0
    while i < len(candidate_labels):
        if candidate_labels[i] == 1:  # 발화 후보군 시작
            start_frame_idx = i
            # 발화 후보군의 끝 찾기
            while i < len(candidate_labels) and candidate_labels[i] == 1:
                i += 1
            
            end_frame_idx = i - 1
            
            # 이 발화 후보군 내에서 90퍼센타일 이상인 값이 하나라도 있는지 확인
            candidate_segment = all_probs[start_frame_idx:end_frame_idx+1]
            if np.any(candidate_segment >= global_threshold_90):
                # 90퍼센타일 이상인 값이 있으면 발화로 간주
                final_labels[start_frame_idx:end_frame_idx+1] = 1
        else:  # 비발화 구간은 그대로 0으로 둠
            i += 1
    
    # 최종 라벨을 사용하여 구간 정보 생성
    current_label = final_labels[0]
    start_frame_idx = 0
    segments_info = []
    
    for i, label in enumerate(final_labels):
        if label == current_label:
            continue
        else:
            segments_info.append((current_label, start_frame_idx, i - 1))
            start_frame_idx = i
            current_label = label
    # 마지막 세그먼트 추가
    segments_info.append((current_label, start_frame_idx, len(final_labels) - 1))
    
    result_dir = os.path.join(out_dir, base_filename)
    os.makedirs(result_dir, exist_ok=True)
    
    # 서브플롯으로 구성된 플롯 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    # 첫 번째 서브플롯: 오디오 파형 및 음성 구간
    time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))
    ax1.plot(time_axis, audio, color='blue', alpha=0.7, label='Audio waveform')
    
    # Speech 구간 표시
    for segment in segments_info:
        label, seg_start_frame, seg_end_frame = segment
        if label == 1:
            # 프레임 인덱스를 시간으로 변환
            seg_start_time = seg_start_frame * frame_rate_ms / 1000
            seg_end_time = (seg_end_frame + 1) * frame_rate_ms / 1000  # +1 to include the end frame
            ax1.axvspan(seg_start_time, seg_end_time, color='red', alpha=0.5, label='Speech')
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Signal with Speech Segments')
    
    # Handle duplicate labels in legend
    handles, labels_legend = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    # 두 번째 서브플롯: VAD 확률값
    # 확률값의 시간 축 생성
    prob_time_axis = np.linspace(0, len(audio) / sample_rate, len(all_probs))
    ax2.plot(prob_time_axis, all_probs, color='green', label='VAD Probability')
    
    # 각 세그먼트의 60 퍼센타일 임계값 표시
    for (start_idx, end_idx), threshold in zip(segment_boundaries, segment_thresholds_60):
        if start_idx >= len(prob_time_axis) or end_idx > len(prob_time_axis):
            continue
        start_time = prob_time_axis[start_idx]
        end_time = prob_time_axis[end_idx - 1]
        ax2.plot([start_time, end_time], [threshold, threshold], 'r--', label='60th percentile')
    
    # 전체 데이터의 90 퍼센타일 임계값 표시
    ax2.axhline(y=global_threshold_90, color='b', linestyle='--', label='90th percentile')
    
    # Speech 구간 표시 (확률 그래프에도 동일하게 표시)
    for segment in segments_info:
        label, seg_start_frame, seg_end_frame = segment
        if label == 1:
            # 프레임 인덱스를 시간으로 변환
            seg_start_time = seg_start_frame * frame_rate_ms / 1000
            seg_end_time = (seg_end_frame + 1) * frame_rate_ms / 1000  # +1 to include the end frame
            ax2.axvspan(seg_start_time, seg_end_time, color='red', alpha=0.3)
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Probability')
    ax2.set_title('VAD Probability with Thresholds')
    
    # 중복된 레이블 처리
    handles, labels_legend = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax2.legend(by_label.values(), by_label.keys())
    
    # 플롯 간격 조정
    plt.tight_layout()
    
    # 플롯을 파일로 저장합니다.
    plot_save_path = os.path.join(result_dir, f"{base_filename}_plot.png")
    plt.savefig(plot_save_path)
    plt.close()

    for i, segment in enumerate(segments_info):
        label, seg_start_frame, seg_end_frame = segment
        
        # 프레임 인덱스를 샘플 인덱스로 변환
        seg_start_sample = seg_start_frame * samples_per_frame
        seg_end_sample = (seg_end_frame + 1) * samples_per_frame  # +1 to include the end frame
        
        # 샘플 인덱스가 오디오 길이를 초과하지 않도록 조정
        seg_end_sample = min(seg_end_sample, len(audio))
        
        if label == 1:
            out_fpath = os.path.join(result_dir, f"{filename}_{i:02d}_speech.wav")
        else:
            out_fpath = os.path.join(result_dir, f"{filename}_{i:02d}_silence.wav")
        
        # 인덱스 범위 검증
        if seg_start_sample >= len(audio) or seg_end_sample > len(audio):
            print(f"Invalid index range: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
            continue
        
        # 오디오 데이터 검증
        segment_audio = audio[seg_start_sample:seg_end_sample]
        if segment_audio.size == 0:
            print(f"Empty audio segment: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
            continue
        
        # 오디오 데이터 값 검증
        if not np.any(segment_audio):
            print(f"Silent audio segment: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
            continue
        
        sf.write(out_fpath, segment_audio, sample_rate)



        
        

def main():
    # SGVAD 모델 초기화 (cfg.yaml 및 ckpt 사용)
    sgvad = SGVAD.init_from_ckpt()
    
    # 결과를 저장할 폴더 생성
    out_dir = "preprocess"
    os.makedirs(out_dir, exist_ok=True)
    
    # "test" 폴더 내 모든 .wav 파일 처리
    wav_files = list(glob.glob("target_resource/*.wav"))[:10]
    if not wav_files:
        print("target_resource 폴더에 처리할 .wav 파일이 없습니다.")
        return
    
    for fpath in wav_files:
        process_audio_file(fpath, sgvad, out_dir)

if __name__ == "__main__":
    main()
import os
from nemo.collections.asr.modules import AudioToMFCCPreprocessor
import numpy as np
import torch
import librosa
import soundfile as sf

device = "cuda"

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

def get_label(mfcc, mfcc_len):
    # MFCC 텐서 (32,-1) -> (-1) 라벨 반환 확률 값임
    script_pt_file = "sgvad_model.pt"
    model_script = torch.jit.load(script_pt_file, map_location=torch.device("cuda"))
    model_script.eval()
    label = model_script(mfcc, mfcc_len)
    return label

def save_audio_segments(segments_info, audio,audio_path):
    for i, segment in enumerate(segments_info):
            samples_per_frame = 16000 // 100
            label, seg_start_frame, seg_end_frame = segment
            
            seg_start_sample = seg_start_frame * samples_per_frame
            seg_end_sample = (seg_end_frame + 1) * samples_per_frame
            seg_end_sample = min(seg_end_sample, len(audio))
            filename = os.path.basename(audio_path).split(".")[0]
            os.makedirs(audio_path.replace(".wav",""), exist_ok=True)
            result_dir = audio_path.replace(".wav","")
            if label == 1:
                out_fpath = os.path.join(result_dir, f"{filename}_{i:02d}_speech.wav")
            else:
                out_fpath = os.path.join(result_dir, f"{filename}_{i:02d}_silence.wav")
            
            if seg_start_sample >= len(audio) or seg_end_sample > len(audio):
                print(f"Invalid index range: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
                continue
            
            segment_audio = audio[seg_start_sample:seg_end_sample]
            if segment_audio.size == 0:
                print(f"Empty audio segment: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
                continue
            
            if not np.any(segment_audio):
                print(f"Silent audio segment: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
                continue
            
            sf.write(out_fpath, segment_audio, sample_rate)


def get_segmented_audio(audio, label,audio_path,smooth=15):
    # smooth 홀수이어야함 15~21 권장
    threshold_low = 0.15
    threshold_high = 0.2
    raw_label = label.squeeze(0).cpu().numpy()
    # 노이즈 강건함을 위한 스무딩 작업 
    kernel = np.ones(smooth) / smooth
    if len (raw_label) < smooth:
        label = raw_label
    else:
        label = np.convolve(raw_label, kernel, mode='same')

    candidate_labels = (label > threshold_low).astype(int)
    final_labels = np.zeros_like(candidate_labels)
    # 후보 구간 중 높은 임계값 이상인 확률값을 포함하는 구간만 발화로 인정
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
    
    for i, label in enumerate(final_labels):
        if label == current_label:
            continue
        else:
            segments_info.append((current_label, start_frame_idx, i - 1))
            start_frame_idx = i
            current_label = label
    segments_info.append((current_label, start_frame_idx, len(final_labels) - 1))
    save_audio_segments(segments_info, audio,audio_path)


if __name__ == "__main__":
    audio_path = "target_wav/비발화_발화소음.wav"
    sample_rate = 16000
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    wave_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    wave_len = torch.tensor([wave_tensor.size(-1)], dtype=torch.long).to(device)
    mfcc, mfcc_len = get_mfcc(wave_tensor, wave_len)
    label = get_label(mfcc, mfcc_len)
    get_segmented_audio(audio, label,audio_path)



    
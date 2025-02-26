import os
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sgvad import SGVAD
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Optional, Dict, Any


class AudioSegmenter:
    """오디오 파일을 세그먼트로 분할하는 클래스"""
    
    def __init__(self, sample_rate: int):
        """
        초기화 함수
        
        Args:
            sample_rate: 오디오 샘플링 주파수 (Hz)
        """
        self.sample_rate = sample_rate
        
    def split_audio_segments(self, audio: np.ndarray, seg_duration: int = 30, 
                             merge_threshold: int = 10) -> List[np.ndarray]:
        """
        오디오를 지정된 길이의 세그먼트로 분할
        
        Args:
            audio: 오디오 신호 배열
            seg_duration: 세그먼트 길이 (초)
            merge_threshold: 병합 임계값 (초)
            
        Returns:
            분할된 오디오 세그먼트 리스트
        """
        seg_samples = int(self.sample_rate * seg_duration)
        merge_samples = int(self.sample_rate * merge_threshold)
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
                    segments.append(np.concatenate((prev_segment, audio[start:total_samples])))
                else:
                    segments.append(audio[start:total_samples])
                break
            else:
                segments.append(audio[start:end])
                start = end
                
        return segments


class VADVisualizer:
    """VAD 결과를 시각화하는 클래스"""
    
    @staticmethod
    def plot_vad_results(audio: np.ndarray, sample_rate: int, all_probs: np.ndarray, 
                         segment_boundaries: List[Tuple[int, int]], 
                         segment_thresholds_60: List[float],
                         sectors_info: List[Tuple[int, int, int]], 
                         frame_rate_ms: int, global_threshold_90: float,
                         plot_save_path: str) -> None:
        """
        오디오 파형과 VAD 확률값을 서브플롯으로 그리는 함수
        
        Args:
            audio: 오디오 신호 (Amplitude) 배열
            sample_rate: 오디오 샘플링 주파수
            all_probs: VAD 예측 확률값 배열
            segment_boundaries: 각 세그먼트의 프레임 인덱스 범위를 담은 리스트, [start, end]
            segment_thresholds_60: 각 세그먼트의 60번째 퍼센타일 임계값 리스트
            sectors_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 리스트
            frame_rate_ms: 프레임 간 시간 간격 (밀리초)
            global_threshold_90: 전체 데이터에 대한 90번째 퍼센타일 값
            plot_save_path: 생성된 플롯을 저장할 파일 경로
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
        
        # 첫 번째 서브플롯: 오디오 파형 및 발화 구간
        time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))
        ax1.plot(time_axis, audio, color='blue', alpha=0.7, label='Audio waveform')
        
        for segment in sectors_info:
            label, seg_start_frame, seg_end_frame = segment
            if label == 1:
                seg_start_time = seg_start_frame * frame_rate_ms / 1000
                seg_end_time = (seg_end_frame + 1) * frame_rate_ms / 1000
                ax1.axvspan(seg_start_time, seg_end_time, color='red', alpha=0.5, label='Speech')
        
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Audio Signal with Speech Segments')
        handles, labels_legend = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax1.legend(by_label.values(), by_label.keys())
        
        # 두 번째 서브플롯: VAD 확률값 및 임계값 표시
        prob_time_axis = np.linspace(0, len(audio) / sample_rate, len(all_probs))
        ax2.plot(prob_time_axis, all_probs, color='green', label='VAD Probability')
        
        for (start_idx, end_idx), threshold in zip(segment_boundaries, segment_thresholds_60):
            if start_idx >= len(prob_time_axis) or end_idx > len(prob_time_axis):
                continue
            start_time = prob_time_axis[start_idx]
            end_time = prob_time_axis[end_idx - 1]
            ax2.plot([start_time, end_time], [threshold, threshold], 'r--', label='60th percentile')
        
        ax2.axhline(y=global_threshold_90, color='b', linestyle='--', label='90th percentile')
        
        for segment in sectors_info:
            label, seg_start_frame, seg_end_frame = segment
            if label == 1:
                seg_start_time = seg_start_frame * frame_rate_ms / 1000
                seg_end_time = (seg_end_frame + 1) * frame_rate_ms / 1000
                ax2.axvspan(seg_start_time, seg_end_time, color='red', alpha=0.3)
        
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Probability')
        ax2.set_title('VAD Probability with Thresholds')
        handles, labels_legend = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax2.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.savefig(plot_save_path)
        plt.close()


class VADProcessor:
    """Voice Activity Detection 처리 클래스"""
    
    def __init__(self, sgvad_model: SGVAD):
        """
        초기화 함수
        
        Args:
            sgvad_model: SGVAD 모델 인스턴스
        """
        self.model = sgvad_model
        self.sample_rate = sgvad_model.cfg.sample_rate
        self.frame_rate_ms = 10  # 10ms per frame
        self.samples_per_frame = int(self.frame_rate_ms / 1000 * self.sample_rate)
        self.segmenter = AudioSegmenter(self.sample_rate)
        self.visualizer = VADVisualizer()
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        오디오 파일 로드
        
        Args:
            file_path: 오디오 파일 경로
            
        Returns:
            로드된 오디오 데이터
        """
        audio = self.model.load_audio(file_path)
        return np.array(audio)
    
    def process_segments(self, segments: List[np.ndarray], smoothing_kernel: int) -> Tuple[
            np.ndarray, np.ndarray, List[Tuple[int, int]], List[float]]:
        """
        오디오 세그먼트 처리 및 VAD 예측
        
        Args:
            segments: 오디오 세그먼트 리스트
            smoothing_kernel: 스무딩 커널 크기
            
        Returns:
            all_probs: 모든 세그먼트의 확률값
            candidate_labels: 60퍼센타일 기준 발화 후보 라벨
            segment_boundaries: 세그먼트별 프레임 경계
            segment_thresholds_60: 각 세그먼트의 60퍼센타일 임계값
        """
        candidate_labels = []  # 60퍼센타일 기준 발화 후보 라벨
        all_probs = []         # 모든 세그먼트의 확률값 저장
        segment_boundaries = []    # 세그먼트별 프레임 경계 저장
        segment_thresholds_60 = []  # 각 세그먼트의 60퍼센타일 임계값 저장
        
        current_position = 0
        for segment in segments:
            probs = self.model.predict(segment, smoothing_kernel)  # (frame_num, 2)
            probs = np.array(probs).ravel()
            
            segment_length = len(probs)
            segment_boundaries.append((current_position, current_position + segment_length))
            threshold_60 = np.percentile(probs, 60)
            segment_thresholds_60.append(threshold_60)
            current_position += segment_length
            
            all_probs.extend(probs.tolist())
            binary_label = (probs > threshold_60).astype(int)
            candidate_labels.extend(binary_label.tolist())
        
        return np.array(candidate_labels), np.array(all_probs), segment_boundaries, segment_thresholds_60
    
    def identify_speech_segments(self, candidate_labels: np.ndarray, all_probs: np.ndarray) -> Tuple[
            np.ndarray, float, List[Tuple[int, int, int]]]:
        """
        발화 구간 식별
        
        Args:
            candidate_labels: 60퍼센타일 기준 발화 후보 라벨
            all_probs: 모든 세그먼트의 확률값
            
        Returns:
            final_labels: 최종 결정된 발화 라벨
            global_threshold_90: 전체 데이터에 대한 90번째 퍼센타일 값
            segments_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 리스트
        """
        global_threshold_90 = np.percentile(all_probs, 90)
        final_labels = np.zeros_like(candidate_labels)
        
        i = 0
        while i < len(candidate_labels):
            if candidate_labels[i] == 1:
                start_frame_idx = i
                while i < len(candidate_labels) and candidate_labels[i] == 1:
                    i += 1
                end_frame_idx = i - 1
                candidate_segment = all_probs[start_frame_idx:end_frame_idx+1]
                if np.any(candidate_segment >= global_threshold_90):
                    final_labels[start_frame_idx:end_frame_idx+1] = 1
            else:
                i += 1
        
        # 세그먼트 정보 생성
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
        
        return final_labels, global_threshold_90, segments_info
    
    def save_audio_segments(self, segments_info: List[Tuple[int, int, int]], 
                          audio: np.ndarray, result_dir: str, filename: str) -> None:
        """
        오디오 세그먼트 저장
        
        Args:
            segments_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 리스트
            audio: 원본 오디오 데이터
            result_dir: 결과 저장 디렉토리
            filename: 파일 이름
        """
        for i, segment in enumerate(segments_info):
            label, seg_start_frame, seg_end_frame = segment
            
            seg_start_sample = seg_start_frame * self.samples_per_frame
            seg_end_sample = (seg_end_frame + 1) * self.samples_per_frame
            seg_end_sample = min(seg_end_sample, len(audio))
            
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
            
            sf.write(out_fpath, segment_audio, self.sample_rate)
    
    def process_audio_file(self, file_path: str, output_dir: str, seg_duration: int = 30, 
                         merge_threshold: int = 10, smoothing_kernel: int = 21, plot: bool = True) -> None:
        """
        오디오 파일 처리 메인 함수
        
        Args:
            file_path: 처리할 오디오 파일 경로
            output_dir: 결과 저장 디렉토리
            seg_duration: 세그먼트 길이 (초)
            merge_threshold: 병합 임계값 (초)
            smoothing_kernel: 스무딩 커널 크기
            plot: 시각화 결과 저장 여부
        """
        try:
            # 오디오 로드
            audio = self.load_audio(file_path)
            filename = os.path.basename(file_path)
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # 오디오 세그먼트 분할
            segments = self.segmenter.split_audio_segments(audio, seg_duration, merge_threshold)
            
            # 세그먼트 처리 및 VAD 예측
            candidate_labels, all_probs, segment_boundaries, segment_thresholds_60 = self.process_segments(
                segments, smoothing_kernel)
            
            # 발화 구간 식별
            final_labels, global_threshold_90, segments_info = self.identify_speech_segments(
                candidate_labels, all_probs)
            
            # 결과 디렉토리 생성
            result_dir = os.path.join(output_dir, base_filename)
            os.makedirs(result_dir, exist_ok=True)
            
            # 시각화 (선택적)
            if plot:
                plot_save_path = os.path.join(result_dir, f"{base_filename}_plot.png")
                self.visualizer.plot_vad_results(
                    audio, 
                    self.sample_rate, 
                    all_probs, 
                    segment_boundaries, 
                    segment_thresholds_60, 
                    segments_info, 
                    self.frame_rate_ms, 
                    global_threshold_90, 
                    plot_save_path
                )
            
            # 오디오 세그먼트 저장
            self.save_audio_segments(segments_info, audio, result_dir, filename)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Voice Activity Detection 프로세서")
    parser.add_argument("--input", default="target_resource/*.wav", help="입력 파일 패턴")
    parser.add_argument("--output", default="preprocess", help="출력 디렉토리")
    parser.add_argument("--max_files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--seg_duration", type=int, default=30, help="세그먼트 길이 (초)")
    parser.add_argument("--merge_threshold", type=int, default=10, help="병합 임계값 (초)")
    parser.add_argument("--smoothing_kernel", type=int, default=21, help="스무딩 커널 크기")
    parser.add_argument("--no_plot", action="store_true", help="시각화 결과 저장 비활성화")
    
    args = parser.parse_args()
    
    # SGVAD 모델 초기화
    try:
        sgvad = SGVAD.init_from_ckpt()
    except Exception as e:
        print(f"SGVAD 모델 초기화 중 오류 발생: {e}")
        return
    
    # VAD 프로세서 초기화
    processor = VADProcessor(sgvad)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 처리할 오디오 파일 목록 가져오기
    wav_files = list(glob.glob(args.input))
    if args.max_files:
        wav_files = wav_files[:args.max_files]
        
    if not wav_files:
        print(f"처리할 .wav 파일이 없습니다: {args.input}")
        return
    
    print(f"총 {len(wav_files)}개 파일 처리 시작...")
    
    # 각 파일 처리
    for idx, fpath in tqdm(enumerate(wav_files)):
        try:
            processor.process_audio_file(
                fpath, 
                args.output,
                seg_duration=args.seg_duration,
                merge_threshold=args.merge_threshold,
                smoothing_kernel=args.smoothing_kernel,
                plot=not args.no_plot
            )
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {fpath} - {e}")
    
    


if __name__ == "__main__":
    main()
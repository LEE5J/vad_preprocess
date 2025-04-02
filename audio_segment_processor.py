import os
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sgvad import SGVAD
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Optional, Dict, Any


class VADVisualizer:
    """VAD 결과를 시각화하는 클래스"""
    
    @staticmethod
    def plot_vad_results(audio: np.ndarray, sample_rate: int, probs: np.ndarray, 
                         sectors_info: List[Tuple[int, int, int]], 
                         frame_rate_ms: int, threshold_low: float, threshold_high: float,
                         plot_save_path: str) -> None:
        """
        오디오 파형과 VAD 확률값을 서브플롯으로 그리는 함수
        
        Args:
            audio: 오디오 신호 (Amplitude) 배열
            sample_rate: 오디오 샘플링 주파수
            probs: VAD 예측 확률값 배열
            sectors_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 리스트
            frame_rate_ms: 프레임 간 시간 간격 (밀리초)
            threshold_low: 낮은 임계값 (0.15)
            threshold_high: 높은 임계값 (0.2)
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
        prob_time_axis = np.linspace(0, len(audio) / sample_rate, len(probs))
        ax2.plot(prob_time_axis, probs, color='green', label='VAD Probability')
        
        ax2.axhline(y=threshold_low, color='r', linestyle='--', label='Low threshold (0.15)')
        ax2.axhline(y=threshold_high, color='b', linestyle='--', label='High threshold (0.2)')
        
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
        self.visualizer = VADVisualizer()
        self.threshold_low = 0.15  # 고정 임계값
        self.threshold_high = 0.2  # 고정 임계값
    
    def identify_speech_segments(self, probs: np.ndarray) -> Tuple[
            np.ndarray, List[Tuple[int, int, int]]]:
        """
        발화 구간 식별
        
        Args:
            probs: VAD 예측 확률값
            
        Returns:
            final_labels: 최종 결정된 발화 라벨
            segments_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 리스트
        """
        # 낮은 임계값 이상인 후보 구간 식별
        candidate_labels = (probs > self.threshold_low).astype(int)
        final_labels = np.zeros_like(candidate_labels)
        
        # 후보 구간 중 높은 임계값 이상인 확률값을 포함하는 구간만 발화로 인정
        i = 0
        while i < len(candidate_labels):
            if candidate_labels[i] == 1:
                start_frame_idx = i
                while i < len(candidate_labels) and candidate_labels[i] == 1:
                    i += 1
                end_frame_idx = i - 1
                candidate_segment = probs[start_frame_idx:end_frame_idx+1]
                if np.any(candidate_segment >= self.threshold_high):
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
        
        return final_labels, segments_info
    
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
    
    def process_audio_files(self, file_paths: List[str], output_dir: str,
                          smoothing_kernel: int = 21, plot: bool = True) -> None:
        """
        여러 오디오 파일을 처리하는 함수 (모든 파일을 한 번에 모델에 전달)
        
        Args:
            file_paths: 처리할 오디오 파일 경로 리스트
            output_dir: 결과 저장 디렉토리
            smoothing_kernel: 스무딩 커널 크기
            plot: 시각화 결과 저장 여부
        """
        if not file_paths:
            print("No audio files to process.")
            return
            
        print(f"Processing {len(file_paths)} audio files...")
        
        # 모든 파일을 한 번에 모델에 전달하여 VAD 예측 수행
        all_probs = self.model.predict(file_paths, smooth=smoothing_kernel)
        
        # 각 파일별로 결과 처리
        for file_idx, file_path in enumerate(file_paths):
            try:
                print(f"Processing file {file_idx+1}/{len(file_paths)}: {file_path}")
                
                # 예측 결과 가져오기
                if file_idx >= len(all_probs) or not all_probs[file_idx]:
                    print(f"Warning: No prediction results for {file_path}")
                    continue
                    
                probs = np.array(all_probs[file_idx])
                
                # 오디오 데이터 불러오기
                audio = self.model.load_audio(file_path)
                if audio is None:
                    print(f"Warning: Failed to load audio: {file_path}")
                    continue
                audio = np.array(audio)
                
                # 파일 정보 설정
                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                
                # 발화 구간 식별
                final_labels, segments_info = self.identify_speech_segments(probs)
                
                # 결과 디렉토리 생성
                result_dir = os.path.join(output_dir, base_filename)
                os.makedirs(result_dir, exist_ok=True)
                
                # 시각화 (선택적)
                if plot:
                    plot_save_path = os.path.join(result_dir, f"{base_filename}_plot.png")
                    self.visualizer.plot_vad_results(
                        audio, 
                        self.sample_rate, 
                        probs, 
                        segments_info, 
                        self.frame_rate_ms, 
                        self.threshold_low, 
                        self.threshold_high, 
                        plot_save_path
                    )
                
                # 오디오 세그먼트 저장
                self.save_audio_segments(segments_info, audio, result_dir, base_filename)
                print(f"Completed processing file: {file_path}")
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                
        print("All files processed.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Voice Activity Detection 프로세서")
    parser.add_argument("--input", default="sample_noise/*.wav", help="입력 파일 패턴")
    parser.add_argument("--output", default="preprocess", help="출력 디렉토리")
    parser.add_argument("--max_files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--smoothing_kernel", type=int, default=21, help="스무딩 커널 크기")
    parser.add_argument("--no_plot", action="store_true", help="시각화 결과 저장 비활성화")
    parser.add_argument("--cfg_path", default="./cfg.yaml", help="SGVAD 설정 파일 경로")
    
    args = parser.parse_args()
    
    # SGVAD 모델 초기화
    try:
        sgvad = SGVAD(cfg_path=args.cfg_path)
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
    
    # 모든 파일을 한 번에 처리
    processor.process_audio_files(
        wav_files,
        args.output,
        smoothing_kernel=args.smoothing_kernel,
        plot=not args.no_plot
    )


if __name__ == "__main__":
    main()
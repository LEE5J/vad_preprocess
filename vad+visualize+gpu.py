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
import subprocess
import tempfile
from typing import List, Tuple, Optional, Dict, Any, Union
import multiprocessing as mp
from functools import partial
import math
import time
import logging


class AudioVisualizer:
    """오디오 시각화를 담당하는 클래스 - 정적 이미지 또는 동영상 생성 지원"""
    
    @staticmethod
    def create_waveform_plot(
            audio: np.ndarray, 
            sample_rate: int, 
            all_probs: np.ndarray = None, 
            segment_boundaries: List[Tuple[int, int]] = None, 
            segment_thresholds_60: List[float] = None,
            sectors_info: List[Tuple[int, int, int]] = None, 
            frame_rate_ms: int = 10, 
            global_threshold_90: float = None,
            figsize: Tuple[int, int] = (12, 8),
            dpi: int = 100
        ) -> Tuple[plt.Figure, Tuple[float, float, float, float]]:
        """
        오디오 파형과 VAD 확률값 그래프를 생성하는 함수
        
        Args:
            audio: 오디오 신호 배열
            sample_rate: 오디오 샘플링 주파수
            all_probs: VAD 예측 확률값 배열 (선택적)
            segment_boundaries: 각 세그먼트의 프레임 인덱스 범위 (선택적)
            segment_thresholds_60: 각 세그먼트의 60번째 퍼센타일 임계값 (선택적)
            sectors_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 (선택적)
            frame_rate_ms: 프레임 간 시간 간격 (밀리초)
            global_threshold_90: 전체 데이터에 대한 90번째 퍼센타일 값 (선택적)
            figsize: 그림 크기
            dpi: 해상도
            
        Returns:
            fig: 생성된 matplotlib Figure 객체
            graph_bbox: 그래프 영역의 바운딩 박스 (x, y, width, height)
        """
        if all_probs is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        
        # 첫 번째 서브플롯: 오디오 파형 및 발화 구간
        time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))
        ax1.plot(time_axis, audio, color='blue', alpha=0.7, label='Audio waveform')
        
        if sectors_info is not None:
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
        
        # 두 번째 서브플롯: VAD 확률값 및 임계값 표시 (선택적)
        if all_probs is not None:
            prob_time_axis = np.linspace(0, len(audio) / sample_rate, len(all_probs))
            ax2.plot(prob_time_axis, all_probs, color='green', label='VAD Probability')
            
            if segment_boundaries is not None and segment_thresholds_60 is not None:
                for (start_idx, end_idx), threshold in zip(segment_boundaries, segment_thresholds_60):
                    if start_idx >= len(prob_time_axis) or end_idx > len(prob_time_axis):
                        continue
                    start_time = prob_time_axis[start_idx]
                    end_time = prob_time_axis[end_idx - 1]
                    ax2.plot([start_time, end_time], [threshold, threshold], 'r--', label='60th percentile')
            
            if global_threshold_90 is not None:
                ax2.axhline(y=global_threshold_90, color='b', linestyle='--', label='90th percentile')
            
            if sectors_info is not None:
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
        
        # 그래프 영역의 바운딩 박스 정보 계산
        fig.canvas.draw()
        if all_probs is not None:
            # 두 서브플롯 중 첫 번째 서브플롯의 위치 정보 사용
            bbox = ax1.get_position()
        else:
            bbox = ax1.get_position()
        
        # 바운딩 박스 정보 (x, y, width, height)
        graph_bbox = (bbox.x0, bbox.y0, bbox.width, bbox.height)
        
        return fig, graph_bbox
    
    @staticmethod
    def save_static_plot(fig: plt.Figure, save_path: str, dpi: int = 100) -> None:
        """
        생성된 그림을 정적 이미지로 저장
        
        Args:
            fig: matplotlib Figure 객체
            save_path: 저장할 파일 경로
            dpi: 해상도
        """
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    
    @staticmethod
    def create_audio_visualization_video(
            audio: np.ndarray, 
            sample_rate: int,
            save_path: str,
            all_probs: np.ndarray = None, 
            segment_boundaries: List[Tuple[int, int]] = None, 
            segment_thresholds_60: List[float] = None,
            sectors_info: List[Tuple[int, int, int]] = None, 
            frame_rate_ms: int = 10, 
            global_threshold_90: float = None,
            fps: int = 30,
            use_gpu: bool = True
        ) -> str:
        """
        오디오 파형과 진행 표시줄이 있는 비디오 생성
        
        Args:
            audio: 오디오 신호 배열
            sample_rate: 오디오 샘플링 주파수
            save_path: 저장할 비디오 파일 경로
            all_probs: VAD 예측 확률값 배열 (선택적)
            segment_boundaries: 각 세그먼트의 프레임 인덱스 범위 (선택적)
            segment_thresholds_60: 각 세그먼트의 60번째 퍼센타일 임계값 (선택적)
            sectors_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 (선택적)
            frame_rate_ms: 프레임 간 시간 간격 (밀리초)
            global_threshold_90: 전체 데이터에 대한 90번째 퍼센타일 값 (선택적)
            fps: 비디오 프레임 레이트
            use_gpu: GPU 가속 사용 여부
            
        Returns:
            save_path: 저장된 비디오 파일 경로
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # 파형 베이스 이미지 생성 및 그래프 영역 정보 가져오기
            fig, graph_bbox = AudioVisualizer.create_waveform_plot(
                audio, sample_rate, all_probs, segment_boundaries, 
                segment_thresholds_60, sectors_info, frame_rate_ms, 
                global_threshold_90
            )
            waveform_img = os.path.join(temp_dir, 'waveform_base.png')
            fig.savefig(waveform_img, dpi=100)
            plt.close(fig)
            
            # 오디오 임시 파일 생성
            audio_temp_path = os.path.join(temp_dir, 'temp_audio.wav')
            sf.write(audio_temp_path, audio, sample_rate)
            
            # 비디오 길이 계산
            duration = len(audio) / sample_rate
            
            # 그래프 영역 정보 추출
            x0, y0, width, height = graph_bbox
            
            # 전체 이미지 크기 가져오기
            img_width, img_height = fig.get_size_inches() * 100  # dpi=100
            
            # 그래프 영역 좌표와 크기를 픽셀 단위로 변환
            x0_px = int(x0 * img_width)
            width_px = int(width * img_width)
            
            # FFmpeg 명령 구성 - 출력 콘솔 메시지 억제
            ffmpeg_log_level = "error"  # 에러 메시지만 표시
            
            # FFmpeg 명령 구성
            try:
                # GPU 가속 사용 시도
                video_encoder = 'h264_nvenc' if use_gpu else 'libx264'
                
                # 진행 표시줄을 그리는 필터 복합체 (세로 전체로 확장)
                filter_complex = (
                    f"color=red:s=5x{int(img_height)}[line];"
                    f"[0:v][line]overlay='{x0_px}+t/{duration}*{width_px}:0'[out]"
                )
                
                # FFmpeg 명령 실행
                subprocess.run([
                    'ffmpeg',
                    '-y',
                    '-loop', '1',
                    '-i', waveform_img,
                    '-i', audio_temp_path,
                    '-filter_complex', filter_complex,
                    '-map', '[out]',
                    '-map', '1:a',
                    '-c:v', video_encoder,
                    '-preset', 'fast',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    '-r', str(fps),
                    '-pix_fmt', 'yuv420p',
                    '-threads', str(mp.cpu_count()),
                    '-loglevel', ffmpeg_log_level,
                    save_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                
            except Exception as e:
                logging.error(f"GPU 가속 실패: {e}. CPU 인코딩으로 전환합니다.")
                
                # CPU 인코딩으로 재시도
                try:
                    subprocess.run([
                        'ffmpeg',
                        '-y',
                        '-loop', '1',
                        '-i', waveform_img,
                        '-i', audio_temp_path,
                        '-filter_complex', filter_complex,
                        '-map', '[out]',
                        '-map', '1:a',
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-shortest',
                        '-r', str(fps),
                        '-pix_fmt', 'yuv420p',
                        '-threads', str(mp.cpu_count()),
                        '-loglevel', ffmpeg_log_level,
                        save_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    
                except Exception as e:
                    logging.error(f"비디오 생성 실패: {e}")
                    raise
        
        return save_path


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
    
    def __init__(self):
        """초기화 함수"""
        self.audio_visualizer = AudioVisualizer()
    
    def visualize_vad_results(
            self, 
            audio: np.ndarray, 
            sample_rate: int, 
            all_probs: np.ndarray, 
            segment_boundaries: List[Tuple[int, int]], 
            segment_thresholds_60: List[float],
            sectors_info: List[Tuple[int, int, int]], 
            frame_rate_ms: int, 
            global_threshold_90: float,
            output_path: str,
            video_output: bool = False,
            fps: int = 30,
            use_gpu: bool = True
        ) -> str:
        """
        VAD 결과 시각화 - 정적 이미지 또는 비디오로 저장
        
        Args:
            audio: 오디오 신호 배열
            sample_rate: 오디오 샘플링 주파수
            all_probs: VAD 예측 확률값 배열
            segment_boundaries: 각 세그먼트의 프레임 인덱스 범위
            segment_thresholds_60: 각 세그먼트의 60번째 퍼센타일 임계값 리스트
            sectors_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 리스트
            frame_rate_ms: 프레임 간 시간 간격 (밀리초)
            global_threshold_90: 전체 데이터에 대한 90번째 퍼센타일 값
            output_path: 출력 파일 경로
            video_output: True면 비디오, False면 정적 이미지로 저장
            fps: 비디오 프레임 레이트 (video_output=True인 경우만 사용)
            use_gpu: GPU 가속 사용 여부 (video_output=True인 경우만 사용)
            
        Returns:
            output_path: 저장된 파일 경로
        """
        if video_output:
            # 비디오로 저장
            return self.audio_visualizer.create_audio_visualization_video(
                audio, sample_rate, output_path,
                all_probs, segment_boundaries, segment_thresholds_60,
                sectors_info, frame_rate_ms, global_threshold_90,
                fps, use_gpu
            )
        else:
            # 정적 이미지로 저장
            fig, _ = self.audio_visualizer.create_waveform_plot(
                audio, sample_rate, all_probs, segment_boundaries,
                segment_thresholds_60, sectors_info, frame_rate_ms,
                global_threshold_90
            )
            self.audio_visualizer.save_static_plot(fig, output_path)
            return output_path


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
            candidate_labels: 60퍼센타일 기준 발화 후보 라벨
            all_probs: 모든 세그먼트의 확률값
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
        """
        for i, segment in enumerate(segments_info):
            label, seg_start_frame, seg_end_frame = segment
            
            seg_start_sample = seg_start_frame * self.samples_per_frame
            seg_end_sample = (seg_end_frame + 1) * self.samples_per_frame
            seg_end_sample = min(seg_end_sample, len(audio))
            
            # 파일 경로 설정
            if label == 1:
                out_fpath = os.path.join(result_dir, f"{filename}_{i:02d}_speech.wav")
            else:
                out_fpath = os.path.join(result_dir, f"{filename}_{i:02d}_silence.wav")
            
            if seg_start_sample >= len(audio) or seg_end_sample > len(audio):
                logging.error(f"Invalid index range: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
                continue
            
            segment_audio = audio[seg_start_sample:seg_end_sample]
            if segment_audio.size == 0:
                logging.error(f"Empty audio segment: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
                continue
            
            # 무음 세그먼트는 DEBUG 레벨로 로깅 (ERROR 대신)
            if not np.any(segment_audio):
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Silent audio segment: start_idx={seg_start_sample}, end_idx={seg_end_sample}")
                # 여전히 저장하려면 아래 코드 유지
        
            sf.write(out_fpath, segment_audio, self.sample_rate)
    
    def process_audio_file(self, file_path: str, output_dir: str, seg_duration: int = 30, 
                         merge_threshold: int = 10, smoothing_kernel: int = 21, 
                         video_output: bool = True, fps: int = 30, use_gpu: bool = True) -> None:
        """
        오디오 파일 처리 메인 함수
        
        Args:
            file_path: 처리할 오디오 파일 경로
            output_dir: 결과 저장 디렉토리
            seg_duration: 세그먼트 길이 (초)
            merge_threshold: 병합 임계값 (초)
            smoothing_kernel: 스무딩 커널 크기
            video_output: True면 비디오, False면 정적 이미지로 시각화 결과 저장
            fps: 비디오 프레임 레이트 (video_output=True인 경우만 사용)
            use_gpu: GPU 가속 사용 여부 (video_output=True인 경우만 사용)
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
            
            # 시각화 (비디오 또는 정적 이미지)
            if video_output:
                output_path = os.path.join(result_dir, f"{base_filename}_visualization.mp4")
            else:
                output_path = os.path.join(result_dir, f"{base_filename}_plot.png")
                
            self.visualizer.visualize_vad_results(
                audio, 
                self.sample_rate, 
                all_probs, 
                segment_boundaries, 
                segment_thresholds_60, 
                segments_info, 
                self.frame_rate_ms, 
                global_threshold_90, 
                output_path,
                video_output,
                fps,
                use_gpu
            )
            
            # 오디오 세그먼트 저장
            self.save_audio_segments(segments_info, audio, result_dir, filename)
            return f"완료:{file_path}"
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return f"실패:{file_path}:{e}"


def process_file_batch(file_batch: List[str], output_dir: str, args) -> List[str]:
    """
    배치 단위로 파일을 처리하는 함수 - 하나의 프로세스에서 모델을 한 번만 로드
    
    Args:
        file_batch: 처리할 파일 경로 목록
        output_dir: 결과 저장 디렉토리
        args: 명령줄 인자
        
    Returns:
        처리 결과 메시지 목록
    """
    try:
        # 모델을 한 번만 로드
        sgvad_model = SGVAD.init_from_ckpt()
        processor = VADProcessor(sgvad_model)
        
        results = []
        for file_path in file_batch:
            try:
                result = processor.process_audio_file(
                    file_path, 
                    output_dir,
                    seg_duration=args.seg_duration,
                    merge_threshold=args.merge_threshold,
                    smoothing_kernel=args.smoothing_kernel,
                    video_output=not args.static_plot,
                    fps=args.fps,
                    use_gpu=not args.no_gpu
                )
                results.append(result)
            except Exception as e:
                results.append(f"실패:{file_path}:{e}")
        
        return results
    except Exception as e:
        logging.error(f"배치 처리 초기화 오류: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [f"실패:{f}:배치 초기화 오류" for f in file_batch]


def main():
    """메인 함수"""
    # 로깅 설정 - 에러 메시지만 표시
    logging.basicConfig(level=logging.ERROR)
    
    parser = argparse.ArgumentParser(description="Voice Activity Detection 프로세서")
    parser.add_argument("--input", default="sample_noise/*.wav", help="입력 파일 패턴")
    parser.add_argument("--output", default="preprocess", help="출력 디렉토리")
    parser.add_argument("--max_files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--seg_duration", type=int, default=30, help="세그먼트 길이 (초)")
    parser.add_argument("--merge_threshold", type=int, default=10, help="병합 임계값 (초)")
    parser.add_argument("--smoothing_kernel", type=int, default=21, help="스무딩 커널 크기")
    parser.add_argument("--static_plot", action="store_true", 
                      help="정적 이미지로 시각화 결과 저장 (기본값: 비디오)")
    parser.add_argument("--fps", type=int, default=30, help="비디오 프레임 레이트")
    parser.add_argument("--no_gpu", action="store_true", help="GPU 가속 비활성화")
    parser.add_argument("--workers", "-w", type=int, default=8, 
                      help="병렬 처리에 사용할 최대 작업자 수 (기본값: 8)")
    parser.add_argument("--quiet", "-q", action="store_true", help="진행 표시줄 숨기기")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 처리할 오디오 파일 목록 가져오기
    wav_files = list(glob.glob(args.input))
    if args.max_files:
        wav_files = wav_files[:args.max_files]
        
    if not wav_files:
        print(f"처리할 .wav 파일이 없습니다: {args.input}")
        return
    
    total_files = len(wav_files)
    
    # 최대 작업자 수 설정 (8개 제한)
    max_workers = min(args.workers, 8, mp.cpu_count())
    
    # 파일을 균등하게 나누기
    batches = []
    batch_size = math.ceil(total_files / max_workers)
    
    for i in range(0, total_files, batch_size):
        batches.append(wav_files[i:i+batch_size])
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 각 프로세스당 배치 처리 시작 (각 프로세스는 자신의 배치를 처리)
    with mp.Pool(len(batches)) as pool:
        all_results = pool.starmap(
            process_file_batch, 
            [(batch, args.output, args) for batch in batches])
        
    
    # 경과 시간 계산
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    # 결과 병합 및 요약
    flattened_results = [item for sublist in all_results for item in sublist]
    successful = sum(1 for r in flattened_results if isinstance(r, str) and r.startswith("완료"))
    failed = len(flattened_results) - successful
    
    # 최종 결과 출력
    print(f"\n처리 결과 요약:")
    print(f"총 파일 수: {total_files}개")
    print(f"성공: {successful}개")
    print(f"실패: {failed}개")
    print(f"총 소요 시간: {int(minutes):d}분 {seconds:.1f}초")
    
    # 실패한 파일이 있으면 로그 기록
    if failed > 0:
        error_files = [r.split(':', 1)[1] for r in flattened_results if isinstance(r, str) and r.startswith("실패")]
        with open(os.path.join(args.output, "error_files.log"), "w") as f:
            for error_file in error_files:
                f.write(f"{error_file}\n")
        print(f"실패한 파일 목록이 {os.path.join(args.output, 'error_files.log')}에 저장되었습니다.")


if __name__ == "__main__":
    print("GPU 사용 버전입니다. 소비자용 GPU 기준으로 맞춰져 있습니다. 프로페셔날 GPU를 사용시 -w 스레드수 옵션을 통해서 더 높은 성능 사용가능합니다.")
    print("소비자용 그래픽카드를 사용시 https://github.com/keylase/nvidia-patch?tab=readme-ov-file 이 링크를 참고하여 최대 스레드수를 제한해제 하면 모든 성능을 사용가능합니다.")
    print("소비자용 그래픽카드는 8개의 스레드만 지원하므로 기본적으로 8개의 스레드만 사용합니다.")
    print("a100과 같은 데이터 센터용 GPU는 nvenc 미지원으로 cpu 버전을 사용해주세요")
    logging.getLogger('nemo').setLevel(logging.ERROR)
    # 로깅 설정 - 치명적인(CRITICAL) 오류만 표시
    #logging.basicConfig(level=logging.CRITICAL)
    mp.set_start_method('spawn', force=True)
    main()
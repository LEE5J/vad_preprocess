# matplotlib 만 사용하며 CPU 인코딩만 사용함
import os
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sgvad import SGVAD
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import argparse
import tempfile
from typing import List, Tuple, Optional, Dict, Any, Union
import multiprocessing as mp
import concurrent.futures
from functools import partial
import time
import subprocess


class AudioVisualizer:
    """오디오 시각화를 담당하는 클래스"""
    
    @staticmethod
    def create_audio_animation(
            audio: np.ndarray, 
            sample_rate: int,
            save_path: str,
            all_probs: np.ndarray = None, 
            sectors_info: List[Tuple[int, int, int]] = None, 
            frame_rate_ms: int = 10, 
            global_threshold_90: float = None,
            fps: int = 30,
            dpi: int = 100
        ) -> str:
        """
        오디오 파형과 진행 표시줄이 있는 애니메이션 비디오 생성
        
        Args:
            audio: 오디오 신호 배열
            sample_rate: 오디오 샘플링 주파수
            save_path: 저장할 비디오 파일 경로
            all_probs: VAD 예측 확률값 배열 (선택적)
            sectors_info: (label, start_frame, end_frame) 형식의 세그먼트 정보 (선택적)
            frame_rate_ms: 프레임 간 시간 간격 (밀리초)
            global_threshold_90: 전체 데이터에 대한 90번째 퍼센타일 값 (선택적)
            fps: 비디오 프레임 레이트
            dpi: 해상도
            
        Returns:
            save_path: 저장된 비디오 파일 경로 또는 오류 시 None
        """
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # 오디오 파일 임시 저장
            temp_audio_path = os.path.join(temp_dir, 'temp_audio.wav')
            sf.write(temp_audio_path, audio, sample_rate)
            
            # 오디오 지속 시간 계산
            duration = len(audio) / float(sample_rate)
            
            # 시간 축 생성
            time_array = np.linspace(0, duration, len(audio))
            
            # 확률 시간 축 생성 (확률 배열이 있는 경우)
            if all_probs is not None:
                prob_time_array = np.linspace(0, duration, len(all_probs))
            
            # 그림 및 축 설정
            if all_probs is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
            
            # 오디오 파형 그리기
            ax1.plot(time_array, audio, color='blue', alpha=0.7, label='Audio waveform')
            
            # 발화 구간 표시 (sectors_info가 있는 경우)
            if sectors_info is not None:
                for segment in sectors_info:
                    label, seg_start_frame, seg_end_frame = segment
                    if label == 1:  # 발화 구간인 경우만 표시
                        seg_start_time = seg_start_frame * frame_rate_ms / 1000
                        seg_end_time = (seg_end_frame + 1) * frame_rate_ms / 1000
                        ax1.axvspan(seg_start_time, seg_end_time, color='red', alpha=0.3, label='Speech')
            
            # 파형 그래프 설정
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Audio Signal with Speech Segments')
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys())
            ax1.set_xlim(0, duration)
            
            # 진행 표시줄 초기화
            progress_line1, = ax1.plot([], [], 'r-', linewidth=2, label='Progress')
            
            # 확률 그래프 설정 (all_probs가 있는 경우)
            if all_probs is not None:
                ax2.plot(prob_time_array, all_probs, color='green', label='VAD Probability')
                
                # 임계값 표시
                if global_threshold_90 is not None:
                    ax2.axhline(y=global_threshold_90, color='b', linestyle='--', label='90th percentile')
                
                # 발화 구간 표시 (sectors_info가 있는 경우)
                if sectors_info is not None:
                    for segment in sectors_info:
                        label, seg_start_frame, seg_end_frame = segment
                        if label == 1:  # 발화 구간인 경우만 표시
                            seg_start_time = seg_start_frame * frame_rate_ms / 1000
                            seg_end_time = (seg_end_frame + 1) * frame_rate_ms / 1000
                            ax2.axvspan(seg_start_time, seg_end_time, color='red', alpha=0.3)
                
                # 확률 그래프 설정
                ax2.set_xlabel('Time [s]')
                ax2.set_ylabel('Probability')
                ax2.set_title('VAD Probability with Thresholds')
                handles, labels = ax2.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax2.legend(by_label.values(), by_label.keys())
                ax2.set_xlim(0, duration)
                ax2.set_ylim(0, 1)
                
                # 진행 표시줄 초기화
                progress_line2, = ax2.plot([], [], 'r-', linewidth=2)
            
            # 애니메이션 업데이트 함수
            def update(frame):
                current_time = frame / fps
                
                # 진행 표시줄 업데이트
                if current_time <= duration:
                    progress_line1.set_data([current_time, current_time], 
                                        [np.min(audio), np.max(audio)])
                    
                    if all_probs is not None:
                        progress_line2.set_data([current_time, current_time], 
                                            [0, 1])
                        return [progress_line1, progress_line2]
                    else:
                        return [progress_line1]
                else:
                    if all_probs is not None:
                        return [progress_line1, progress_line2]
                    else:
                        return [progress_line1]
            
            # 애니메이션 생성
            frames = int(duration * fps)
            ani = FuncAnimation(fig, update, frames=range(frames), blit=True)
            
            plt.tight_layout()
            
            # 비디오 파일 저장
            print(f"애니메이션 저장 중... ({frames} 프레임)")
            
            # 인코더 설정
            temp_video = os.path.join(temp_dir, 'temp_video.mp4')
            
            try:
                # CPU 인코더 설정
                writer_options = {
                    'codec': 'libx264',
                    'fps': fps,
                    'bitrate': 1800,
                    'extra_args': [
                        '-preset', 'fast',  # 인코딩 속도 vs 품질 (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
                        '-crf', '23',       # 품질 수준 (0-51, 낮을수록 고품질)
                        '-pix_fmt', 'yuv420p'  # 더 넓은 플레이어 호환성
                    ]
                }
                
                print("CPU 인코더로 비디오 인코딩 시작")
                ani.save(
                    temp_video, 
                    writer='ffmpeg', 
                    dpi=dpi, 
                    **writer_options
                )
                
                print(f"비디오 인코딩 완료: {temp_video}")
                
                # 오디오 추가
                try:
                    # FFmpeg로 오디오 추가
                    ffmpeg_cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video,
                        '-i', temp_audio_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-strict', 'experimental',
                        '-map', '0:v:0',
                        '-map', '1:a:0',
                        '-shortest',
                        save_path
                    ]
                    
                    print("오디오 추가 중...")
                    process = subprocess.run(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    
                    print(f"최종 비디오 저장 완료: {save_path}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"오디오 추가 실패: {e}")
                    print(f"오류 메시지: {e.stderr}")
                    print("오디오 없이 비디오만 저장합니다.")
                    import shutil
                    shutil.copy(temp_video, save_path)
                    
            except Exception as e:
                print(f"비디오 인코딩 실패: {e}")
                
                # 더 낮은 품질로 재시도
                try:
                    print("인코딩 실패, 더 낮은 품질로 재시도합니다...")
                    writer_options = {
                        'codec': 'libx264',
                        'fps': fps,
                        'bitrate': 1200,
                        'extra_args': [
                            '-preset', 'ultrafast',  # 최대한 빠른 인코딩
                            '-crf', '28',            # 낮은 품질 (빠른 처리)
                            '-pix_fmt', 'yuv420p'
                        ]
                    }
                    
                    ani.save(
                        temp_video, 
                        writer='ffmpeg', 
                        dpi=dpi, 
                        **writer_options
                    )
                    
                    # 오디오 추가
                    subprocess.run([
                        'ffmpeg', '-y',
                        '-i', temp_video,
                        '-i', temp_audio_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-strict', 'experimental',
                        '-shortest',
                        save_path
                    ], check=True)
                    
                    print(f"낮은 품질로 비디오 저장 성공: {save_path}")
                    
                except Exception as retry_e:
                    print(f"낮은 품질로도 인코딩 실패: {retry_e}")
                    plt.close(fig)
                    return None
            
            # 메모리 정리
            plt.close(fig)
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
        self.visualizer = AudioVisualizer()
    
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
                         merge_threshold: int = 10, smoothing_kernel: int = 21, 
                         fps: int = 30) -> None:
        """
        오디오 파일 처리 메인 함수
        
        Args:
            file_path: 처리할 오디오 파일 경로
            output_dir: 결과 저장 디렉토리
            seg_duration: 세그먼트 길이 (초)
            merge_threshold: 병합 임계값 (초)
            smoothing_kernel: 스무딩 커널 크기
            fps: 비디오 프레임 레이트
        """
        try:
            start_time = time.time()
            
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
            final_labels, global_threshold_90, sectors_info = self.identify_speech_segments(
                candidate_labels, all_probs)
            
            # 결과 디렉토리 생성
            result_dir = os.path.join(output_dir, base_filename)
            os.makedirs(result_dir, exist_ok=True)
            
            # 오디오 세그먼트 저장
            self.save_audio_segments(sectors_info, audio, result_dir, filename)
            
            # 비디오 애니메이션 생성
            output_path = os.path.join(result_dir, f"{base_filename}_visualization.mp4")
            
            self.visualizer.create_audio_animation(
                audio, 
                self.sample_rate, 
                output_path,
                all_probs, 
                sectors_info, 
                self.frame_rate_ms, 
                global_threshold_90, 
                fps,
                100
            )
            
            end_time = time.time()
            print(f"처리 완료. 소요 시간: {end_time - start_time:.2f}초")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            import traceback
            traceback.print_exc()


def process_file_worker(file_path: str, output_dir: str, 
                        seg_duration: int, merge_threshold: int, smoothing_kernel: int,
                        fps: int) -> None:
    """
    개별 파일 처리 워커 함수 (병렬 처리용)
    """
    try:
        print(f"처리 시작: {file_path}")
        # 각 프로세스에서 독립적으로 모델 로드
        sgvad_model = SGVAD.init_from_ckpt()
        processor = VADProcessor(sgvad_model)
        processor.process_audio_file(
            file_path, 
            output_dir, 
            seg_duration, 
            merge_threshold, 
            smoothing_kernel, 
            fps
        )
        print(f"처리 완료: {file_path}")
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {file_path} - {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="병렬 처리 Voice Activity Detection 프로세서")
    parser.add_argument("--input", default="sample_noise/*.wav", help="입력 파일 패턴")
    parser.add_argument("--output", default="preprocess", help="출력 디렉토리")
    parser.add_argument("--max_files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--seg_duration", type=int, default=30, help="세그먼트 길이 (초)")
    parser.add_argument("--merge_threshold", type=int, default=10, help="병합 임계값 (초)")
    parser.add_argument("--smoothing_kernel", type=int, default=21, help="스무딩 커널 크기")
    parser.add_argument("--fps", type=int, default=30, help="비디오 프레임 레이트")
    parser.add_argument("--processes", type=int, default=None, 
                      help="병렬 처리에 사용할 프로세스 수 (기본값: CPU 코어 수)")
    
    args = parser.parse_args()
    
    # 프로세스 수 설정
    num_processes = args.processes if args.processes is not None else max(1, mp.cpu_count() - 1)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 처리할 오디오 파일 목록 가져오기
    wav_files = list(glob.glob(args.input))
    if args.max_files:
        wav_files = wav_files[:args.max_files]
        
    if not wav_files:
        print(f"처리할 .wav 파일이 없습니다: {args.input}")
        return
    
    print(f"총 {len(wav_files)}개 파일을 {num_processes}개 프로세스로 병렬 처리합니다...")
    print(f"CPU 인코딩 모드로 실행 중...")
    
    # 병렬 처리 실행
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 각 작업에 필요한 인자들을 직접 전달 (모델 공유 X)
        futures = [
            executor.submit(
                process_file_worker,
                file_path=fpath,
                output_dir=args.output,
                seg_duration=args.seg_duration,
                merge_threshold=args.merge_threshold,
                smoothing_kernel=args.smoothing_kernel,
                fps=args.fps
            ) for fpath in wav_files
        ]
        
        # 결과 처리
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                future.result()  # 워커에서 발생한 예외 전파
                print(f"진행 상황: {i+1}/{len(wav_files)} 완료")
            except Exception as e:
                print(f"작업 실패: {e}")
                import traceback
                print(f"오류 상세: {traceback.format_exc()}")
    
    end_time = time.time()
    print(f"모든 처리 완료. 총 소요 시간: {end_time - start_time:.2f}초")


if __name__ == "__main__":
    main()
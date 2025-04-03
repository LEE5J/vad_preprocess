import os
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import subprocess
import glob
import tempfile
import shutil
from typing import Tuple, List, Optional
import json
from sgvad import SGVAD  # SGVAD 클래스 임포트


def create_progress_frame(base_image: Image.Image, progress_x: int, 
                         line_color: Tuple[int, int, int] = (255, 0, 0), 
                         line_width: int = 2,
                         vad_data: Optional[List[float]] = None,
                         vad_progress_idx: Optional[int] = None,
                         vad_color: Tuple[int, int, int] = (0, 255, 0),
                         vad_threshold: float = 0.5) -> Image.Image:
    """
    지정된 x 위치에 수직 진행 선을 가진 프레임을 생성하고 선택적으로 VAD 시각화 포함
    
    Args:
        base_image: 그릴 기본 이미지
        progress_x: 진행 선의 X 좌표
        line_color: 선의 RGB 색상 튜플
        line_width: 진행 선의 너비
        vad_data: 선택적 VAD 점수 리스트
        vad_progress_idx: progress_x에 해당하는 VAD 데이터의 현재 인덱스
        vad_color: VAD 시각화를 위한 RGB 색상 튜플
        vad_threshold: VAD 활동 감지를 위한 임계값
    
    Returns:
        진행 선이 있는 프레임(PIL Image)
    """
    # 원본 이미지 수정 방지를 위해 복사본 생성
    frame = base_image.copy()
    draw = ImageDraw.Draw(frame)
    
    # progress_x 위치에 수직선 그리기
    draw.line([(progress_x, 0), (progress_x, base_image.height)], 
             fill=line_color, width=line_width)
    
    # VAD 데이터가 제공된 경우 시각화 추가
    if vad_data is not None and vad_progress_idx is not None:
        # 현재 VAD 상태 강조
        if 0 <= vad_progress_idx < len(vad_data):
            vad_score = vad_data[vad_progress_idx]
            # 프레임 하단에 VAD 인디케이터 그리기
            indicator_height = 20
            indicator_y = base_image.height - indicator_height
            
            # VAD 점수에 따라 색상 강도 결정
            alpha = min(1.0, vad_score / vad_threshold) if vad_threshold > 0 else vad_score
            color_intensity = int(255 * alpha) if vad_score >= vad_threshold else 0
            
            # VAD 인디케이터 그리기
            if vad_score >= vad_threshold:
                draw.rectangle([progress_x-5, indicator_y, progress_x+5, base_image.height], 
                              fill=(0, color_intensity, 0))
    
    return frame


def process_audio_with_vad(audio_paths: List[str]) -> dict:
    """
    SGVAD 모델을 사용하여 오디오 파일 목록 처리
    
    Args:
        audio_paths: 오디오 파일 경로 목록
    
    Returns:
        오디오 파일 경로를 키로, VAD 점수를 값으로 하는 사전
    """
    try:
        # SGVAD 초기화
        sgvad = SGVAD()
        
        # 모든 오디오 파일 한 번에 처리
        vad_scores = sgvad.predict(audio_paths)
        
        # 파일 경로를 점수에 매핑하는 사전 생성
        results = {path: scores for path, scores in zip(audio_paths, vad_scores)}
        
        return results
    except Exception as e:
        print(f"VAD로 오디오 처리 중 오류 발생: {e}")
        return {}


def generate_video_with_progress(image_path: str, audio_path: str, 
                               display_start: int, display_end: int, 
                               output_path: str, fps: int = 30, 
                               line_color: Tuple[int, int, int] = (255, 0, 0),
                               line_width: int = 2,
                               vad_data: Optional[List[float]] = None,
                               vad_color: Tuple[int, int, int] = (0, 255, 0),
                               vad_threshold: float = 0.5) -> bool:
    """
    오디오와 동기화된 진행 선이 있는 비디오 생성
    
    Args:
        image_path: 시각화 이미지 경로
        audio_path: 오디오 파일 경로
        display_start: 진행 선의 시작 x 좌표(픽셀)
        display_end: 진행 선의 끝 x 좌표(픽셀)
        output_path: 출력 비디오 저장 경로
        fps: 출력 비디오의 초당 프레임 수
        line_color: 진행 선의 RGB 색상 튜플
        line_width: 진행 선의 너비
        vad_data: 시각화를 위한 선택적 VAD 점수 리스트
        vad_color: VAD 시각화를 위한 RGB 색상 튜플
        vad_threshold: VAD 활동 감지를 위한 임계값
    
    Returns:
        성공 또는 실패를 나타내는 부울 값
    """
    try:
        # 기본 이미지 로드
        base_image = Image.open(image_path)
        
        # FFprobe를 사용하여 오디오 지속 시간 가져오기
        duration_cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'json', 
            audio_path
        ]
        
        result = subprocess.run(duration_cmd, capture_output=True, text=True)
        duration_data = json.loads(result.stdout)
        audio_duration = float(duration_data['format']['duration'])
        
        # 총 프레임 수 계산
        total_frames = int(audio_duration * fps)
        
        # 각 프레임의 진행 좌표 계산
        progress_range = display_end - display_start
        
        # 프레임 저장을 위한 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # 진행 선이 있는 프레임 생성
            for frame_idx in range(total_frames):
                # 현재 시간 위치 계산
                current_time = frame_idx / fps
                
                # 진행 선의 현재 x 위치 계산
                progress_ratio = current_time / audio_duration
                progress_x = int(display_start + progress_ratio * progress_range)
                
                # 사용 가능한 경우 VAD 인덱스 계산
                vad_idx = None
                if vad_data is not None:
                    vad_idx = min(int(progress_ratio * len(vad_data)), len(vad_data) - 1)
                
                # 진행 선과 VAD 데이터(사용 가능한 경우)로 프레임 생성
                frame = create_progress_frame(
                    base_image, progress_x, line_color, line_width,
                    vad_data=vad_data, vad_progress_idx=vad_idx,
                    vad_color=vad_color, vad_threshold=vad_threshold
                )
                
                # 프레임 저장
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                frame.save(frame_path)
            
            # NVIDIA 하드웨어 가속이 가능한 경우 FFmpeg으로 비디오 생성
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # 출력 파일이 존재하는 경우 덮어쓰기
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-i', audio_path,
                '-c:v', 'h264_nvenc',  # NVIDIA 하드웨어 인코딩
                '-preset', 'slow',
                '-b:v', '5M',  # 비디오 비트레이트
                '-c:a', 'aac',  # 오디오 코덱
                '-b:a', '192k',  # 오디오 비트레이트
                '-shortest',
                output_path
            ]
            
            # 실행 전 NVENC 가용성 확인
            check_cmd = ['ffmpeg', '-hide_banner', '-encoders']
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if 'h264_nvenc' not in check_result.stdout:
                # NVENC를 사용할 수 없는 경우 CPU 인코딩으로 대체
                ffmpeg_cmd[ffmpeg_cmd.index('-c:v') + 1] = 'libx264'
            
            # FFmpeg 명령 조용히 실행
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
            
    except Exception as e:
        # 멀티프로세싱을 위한 조용한 실패
        print(f"비디오 생성 중 오류 발생: {e}")
        return False


def load_coordinates_from_log(log_file: str) -> dict:
    """
    로그 파일에서 표시 좌표 로드
    
    Args:
        log_file: 좌표가 포함된 로그 파일 경로
    
    Returns:
        파일 이름을 키로, 좌표 데이터를 값으로 하는 사전
    """
    coordinates = {}
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith("File:"):
                    parts = line.strip().split(", ")
                    if len(parts) >= 4:
                        filename = parts[0].replace("File: ", "")
                        start = int(parts[1].replace("Start: ", ""))
                        end = int(parts[2].replace("End: ", ""))
                        duration = float(parts[3].replace("Duration: ", "").replace("s", ""))
                        coordinates[filename] = {
                            "start": start,
                            "end": end,
                            "duration": duration
                        }
    except Exception as e:
        print(f"로그 파일에서 좌표 로드 중 오류 발생: {e}")
    
    return coordinates


def batch_process_files(base_dir: str, log_file: Optional[str] = None, fps: int = 30,
                       use_vad: bool = False, vad_cfg_path: Optional[str] = None,
                       vad_threshold: float = 0.5) -> None:
    """
    디렉토리의 모든 시각화 이미지 및 오디오 파일 처리
    
    Args:
        base_dir: 시각화 이미지 및 오디오 파일이 포함된 기본 디렉토리
        log_file: 좌표가 포함된 로그 파일 경로 (None인 경우 기본값 사용)
        fps: 출력 비디오의 초당 프레임 수
        use_vad: VAD를 시각화에 사용할지 여부
        vad_cfg_path: VAD 구성 파일 경로
        vad_threshold: VAD 활동 감지를 위한 임계값
    """
    # 로그 파일이 제공된 경우 좌표 로드
    coordinates = {}
    if log_file and os.path.exists(log_file):
        coordinates = load_coordinates_from_log(log_file)
    
    # 모든 시각화 이미지 찾기
    viz_files = glob.glob(os.path.join(base_dir, "result_compare_plt", "*_comparison.png"))
    
    # 출력 디렉토리 생성
    output_dir = os.path.join(base_dir, "result_videos")
    os.makedirs(output_dir, exist_ok=True)
    
    # 필요한 경우 배치 VAD 처리를 위한 모든 오디오 파일 수집
    audio_files = []
    base_names = []
    
    for viz_file in viz_files:
        # 기본 파일명 추출
        base_name = os.path.basename(viz_file).replace("_comparison.png", "")
        
        # 해당 오디오 파일 찾기
        audio_file = os.path.join(base_dir, "target_audio", f"{base_name}.wav")
        if not os.path.exists(audio_file):
            print(f"{base_name}에 대한 오디오 파일을 찾을 수 없습니다")
            continue
        
        audio_files.append(audio_file)
        base_names.append(base_name)
    
    # VAD를 사용하는 경우 모든 파일을 한 번에 처리
    vad_results = {}
    if use_vad and audio_files:
        print(f"{len(audio_files)}개의 오디오 파일에 대해 VAD 처리 중...")
        vad_results = process_audio_with_vad(audio_files)
    
    # 각 시각화를 해당 오디오 파일과 함께 처리
    for idx, (base_name, viz_file) in enumerate(zip(base_names, viz_files)):
        try:
            audio_file = audio_files[idx]
            
            # 표시 좌표 가져오기
            if base_name in coordinates:
                display_start = coordinates[base_name]["start"]
                display_end = coordinates[base_name]["end"]
            else:
                # 로그 파일에 없는 경우 기본값 사용
                print(f"{base_name}에 대한 좌표를 찾을 수 없습니다. 기본값을 사용합니다")
                display_start = 100
                display_end = 1800
            
            # 출력 비디오 경로
            output_video = os.path.join(output_dir, f"{base_name}_video.mp4")
            
            # 사용 가능한 경우 VAD 데이터 가져오기
            vad_data = vad_results.get(audio_file, None) if use_vad else None
            
            # 비디오 생성
            success = generate_video_with_progress(
                viz_file, audio_file, 
                display_start, display_end, 
                output_video, fps,
                vad_data=vad_data,
                vad_threshold=vad_threshold
            )
            
            if success:
                print(f"{base_name} 처리 완료")
            else:
                print(f"{base_name} 처리 실패")
            
        except Exception as e:
            print(f"{viz_file} 처리 중 오류 발생: {e}")


def main():
    parser = argparse.ArgumentParser(description="NVIDIA 인코딩을 사용한 진행 선 비디오 생성")
    parser.add_argument("--base-dir", default=".", help="처리를 위한 기본 디렉토리")
    parser.add_argument("--log-file", help="좌표가 포함된 로그 파일 경로")
    parser.add_argument("--fps", type=int, default=30, help="초당 프레임 수")
    parser.add_argument("--use-vad", action="store_true", help="시각화에 음성 활동 감지 사용")
    parser.add_argument("--vad-cfg", help="VAD 구성 파일 경로")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD 활동 감지를 위한 임계값")
    
    args = parser.parse_args()
    
    # 항상 배치 처리 수행
    batch_process_files(
        args.base_dir, 
        args.log_file, 
        args.fps,
        use_vad=args.use_vad, 
        vad_cfg_path=args.vad_cfg,
        vad_threshold=args.vad_threshold
    )


if __name__ == "__main__":
    main()

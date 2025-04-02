# vad_visualize_gpu.py
import os
import glob
import numpy as np
import logging
import argparse
import multiprocessing as mp
import math
import time
from typing import List
from sgvad import SGVAD

# 공통 모듈 임포트
from vad_visualize_core import VADProcessorBase


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
        sgvad_model = SGVAD()
        processor = VADProcessorBase(sgvad_model)
        
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
                    use_gpu=True
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
    
    parser = argparse.ArgumentParser(description="Voice Activity Detection 프로세서 (GPU 지원)")
    parser.add_argument("--input", default="target_audio/*.wav", help="입력 파일 패턴")
    parser.add_argument("--output", default="result_audio", help="출력 디렉토리")
    parser.add_argument("--max_files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--seg_duration", type=int, default=30, help="세그먼트 길이 (초)")
    parser.add_argument("--merge_threshold", type=int, default=10, help="병합 임계값 (초)")
    parser.add_argument("--smoothing_kernel", type=int, default=15, help="스무딩 커널 크기")
    parser.add_argument("--static_plot", action="store_true", 
                      help="정적 이미지로 시각화 결과 저장 (기본값: 비디오)")
    parser.add_argument("--fps", type=int, default=30, help="비디오 프레임 레이트")
    parser.add_argument("--workers", "-w", type=int, default=8, 
                      help="병렬 처리에 사용할 최대 작업자 수 (기본값: 8)")
    
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
    print("소비자용 그래픽카드는 8개의 스레드만 지원하므로 기본적으로 8개의 스레드만 사용합니다. 24년 1월 이후 드라이버 버전부터 8개의 스레드 지원")
    print("a100과 같은 데이터 센터용 GPU는 nvenc 미지원으로 cpu 버전을 사용해주세요")
    logging.getLogger('nemo').setLevel(logging.ERROR)
    mp.set_start_method('spawn', force=True)
    main()
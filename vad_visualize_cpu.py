# vad_visualize_cpu.py
import os
import glob
import logging
import argparse
import multiprocessing as mp
import concurrent.futures
import time
from sgvad import SGVAD
from vad_visualize_core import VADProcessorBase



def process_file_worker(file_path: str, output_dir: str, 
                        seg_duration: int, merge_threshold: int, smoothing_kernel: int,
                        fps: int) -> str:
    """
    개별 파일 처리 워커 함수 (병렬 처리용)
    
    Args:
        file_path: 처리할 오디오 파일 경로
        output_dir: 결과 저장 디렉토리
        seg_duration: 세그먼트 길이 (초)
        merge_threshold: 병합 임계값 (초)
        smoothing_kernel: 스무딩 커널 크기
        fps: 비디오 프레임 레이트
        
    Returns:
        처리 결과 메시지
    """
    try:
        # print(f"처리 시작: {file_path}")
        # 각 프로세스에서 독립적으로 모델 로드
        sgvad_model = SGVAD()
        processor = VADProcessorBase(sgvad_model)
        result = processor.process_audio_file(
            file_path, 
            output_dir, 
            seg_duration, 
            merge_threshold, 
            smoothing_kernel, 
            True,  # video_output=True
            fps,
            False  # use_gpu=False
        )
        # print(f"처리 완료: {file_path}")
        return result
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {file_path} - {e}")
        import traceback
        traceback.print_exc()
        return f"실패:{file_path}:{e}"


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Voice Activity Detection 프로세서 (CPU 버전)")
    parser.add_argument("--input", default="target_audio/*.wav", help="입력 파일 패턴")
    parser.add_argument("--output", default="result_audio", help="출력 디렉토리")
    parser.add_argument("--max_files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--seg_duration", type=int, default=30, help="세그먼트 길이 (초)")
    parser.add_argument("--merge_threshold", type=int, default=10, help="병합 임계값 (초)")
    parser.add_argument("--smoothing_kernel", type=int, default=21, help="스무딩 커널 크기")
    parser.add_argument("--fps", type=int, default=10, help="비디오 프레임 레이트")
    parser.add_argument("--workers", "-w", type=int, default=None, 
                      help="병렬 처리에 사용할 프로세스 수 (기본값: CPU 코어 수)")
    
    args = parser.parse_args()
    
    # 프로세스 수 설정
    num_processes = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    
    
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
    num_processes = min(num_processes,total_files)
    print(f"총 {total_files}개 파일을 {num_processes}개 프로세스로 병렬 처리합니다...")
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
        
        # 결과 처리 (GPU 버전과 동일한 결과 형식 유지)
        all_results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                all_results.append(result)
                print(f"진행 상황: {i+1}/{total_files} 완료")
            except Exception as e:
                print(f"작업 실패: {e}")
                all_results.append(f"실패:알 수 없는 오류:{e}")
    
    # 경과 시간 계산
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    # 결과 요약 (GPU 버전과 동일한 출력 형식 유지)
    successful = sum(1 for r in all_results if isinstance(r, str) and r.startswith("완료"))
    failed = len(all_results) - successful
    
    # 최종 결과 출력
    print(f"\n처리 결과 요약:")
    print(f"총 파일 수: {total_files}개")
    print(f"성공: {successful}개")
    print(f"실패: {failed}개")
    print(f"총 소요 시간: {int(minutes):d}분 {seconds:.1f}초")
    
    # 실패한 파일이 있으면 로그 기록
    if failed > 0:
        error_files = [r.split(':', 1)[1] for r in all_results if isinstance(r, str) and r.startswith("실패")]
        with open(os.path.join(args.output, "error_files.log"), "w") as f:
            for error_file in error_files:
                f.write(f"{error_file}\n")
        print(f"실패한 파일 목록이 {os.path.join(args.output, 'error_files.log')}에 저장되었습니다.")


if __name__ == "__main__":
    print("CPU 전용 버전입니다.")
    print("Matplotlib 애니메이션을 사용한 인코딩을 수행합니다.")
    mp.set_start_method('spawn', force=True)
    main()
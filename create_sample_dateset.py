import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time  # 시간 측정을 위해 추가

# --- 설정 ---
# 입력 폴더 경로
label_tg_dir = Path("label_tg")  # .TextGrid 파일이 있는 폴더
noise_mixed_audio_dir = Path("noise_mixed_audio") # .wav 파일이 있는 폴더 (하위 폴더 포함)

# 출력 폴더 경로
sample_tg_dir = Path("sample_tg")
sample_noise_mixed_audio_dir = Path("sample_noise_mixed_audio")

# 샘플링할 파일 개수
num_samples = 5000

# 재현성을 위한 랜덤 시드
random_seed = 42

# 사용할 프로세스 수 (None으로 설정하면 os.cpu_count() 사용)
# 디스크 I/O 바운드 작업이므로 CPU 코어 수보다 약간 많게 설정하는 것이 유리할 수 있으나,
# 너무 많으면 오히려 느려질 수 있습니다. None이나 CPU 코어 수로 시작하는 것을 권장합니다.
# 예: num_workers = os.cpu_count() or num_workers = os.cpu_count() + 2
num_workers = os.cpu_count()
# --- 설정 끝 ---


def process_single_textgrid(tg_source_path, noise_mixed_audio_path, sample_tg_path, sample_noise_mixed_audio_path):
    """
    하나의 TextGrid 파일과 그에 해당하는 모든 오디오 파일을 처리하는 함수 (단일 프로세스 작업 단위).

    Args:
        tg_source_path (Path): 처리할 원본 TextGrid 파일 경로.
        noise_mixed_audio_path (Path): 원본 오디오 파일 루트 경로.
        sample_tg_path (Path): 샘플 TextGrid 파일을 저장할 경로.
        sample_noise_mixed_audio_path (Path): 샘플 오디오 파일을 저장할 경로.

    Returns:
        tuple: (tg_copied_path or None, num_wav_copied, wav_found_flag, error_message or None)
               - tg_copied_path: 성공적으로 복사된 TextGrid 파일 경로 (실패 시 None)
               - num_wav_copied: 성공적으로 복사된 WAV 파일 수
               - wav_found_flag: 해당하는 WAV 파일을 하나라도 찾았는지 여부 (True/False)
               - error_message: 처리 중 발생한 오류 메시지 (오류 없을 시 None)
    """
    tg_copied_path = None
    num_wav_copied = 0
    wav_found_flag = False
    error_message = None

    try:
        # --- 1. TextGrid 파일 복사 ---
        tg_filename = tg_source_path.name
        tg_dest_path = sample_tg_path / tg_filename
        # 대상 폴더가 없을 경우를 대비 (여러 프로세스가 동시에 생성 시도 가능)
        sample_tg_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tg_source_path, tg_dest_path)
        tg_copied_path = tg_dest_path # 성공 시 경로 저장

        # --- 2. 해당하는 모든 오디오(.wav) 파일 찾기 및 복사 ---
        base_filename = tg_source_path.stem
        wav_filename = f"{base_filename}.wav"

        # noise_mixed_audio 폴더 및 하위 폴더에서 해당 .wav 파일 재귀적으로 검색
        found_wav_paths = list(noise_mixed_audio_path.rglob(f"**/{wav_filename}"))

        if found_wav_paths:
            wav_found_flag = True
            for wav_source_path in found_wav_paths:
                try:
                    # 원본 오디오 파일의 상대 경로 계산
                    relative_wav_dir = wav_source_path.parent.relative_to(noise_mixed_audio_path)

                    # 샘플 오디오 폴더 내에 동일한 하위 폴더 구조 생성
                    wav_dest_dir = sample_noise_mixed_audio_path / relative_wav_dir
                    wav_dest_dir.mkdir(parents=True, exist_ok=True) # 동시성 문제 방지

                    # 최종 오디오 파일 대상 경로
                    wav_dest_path = wav_dest_dir / wav_filename

                    # 오디오 파일 복사
                    shutil.copy2(wav_source_path, wav_dest_path)
                    num_wav_copied += 1

                except Exception as e_inner:
                    # 개별 WAV 파일 복사 오류는 기록하되 전체 프로세스를 중단시키지는 않음
                    print(f"\n경고: 오디오 파일 '{wav_source_path}' 복사 중 오류: {e_inner} (대상: {wav_dest_path})")
                    # 필요하다면 에러 메시지를 누적할 수도 있음
                    if error_message is None:
                        error_message = f"WAV copy error: {e_inner}"
                    else:
                        error_message += f"; WAV copy error: {e_inner}"
        # else: wav_found_flag는 False로 유지됨

    except Exception as e_outer:
        error_message = f"Error processing TG '{tg_source_path.name}': {e_outer}"
        # TextGrid 복사 실패 시 tg_copied_path는 None으로 유지됨

    return tg_copied_path, num_wav_copied, wav_found_flag, error_message


def create_sample_dataset_parallel(label_tg_path, noise_mixed_audio_path, sample_tg_path, sample_noise_mixed_audio_path, n_samples, seed, max_workers):
    """
    멀티프로세싱을 사용하여 샘플 데이터셋을 생성합니다.

    Args:
        label_tg_path (Path): 원본 TextGrid 파일 경로.
        noise_mixed_audio_path (Path): 원본 오디오 파일 경로 (하위 폴더 포함).
        sample_tg_path (Path): 샘플링된 TextGrid 파일을 저장할 경로.
        sample_noise_mixed_audio_path (Path): 샘플링된 오디오 파일을 저장할 경로.
        n_samples (int): 샘플링할 TextGrid 파일 개수.
        seed (int): 랜덤 시드 값.
        max_workers (int or None): 사용할 최대 프로세스 수.
    """
    start_time = time.time()
    print("스크립트 시작 (병렬 처리)...")
    print(f"원본 TextGrid 폴더: {label_tg_path}")
    print(f"원본 오디오 폴더: {noise_mixed_audio_path}")
    print(f"샘플 TextGrid 저장 폴더: {sample_tg_path}")
    print(f"샘플 오디오 저장 폴더: {sample_noise_mixed_audio_path}")
    print(f"샘플링할 TextGrid 개수: {n_samples}")
    print(f"랜덤 시드: {seed}")
    print(f"사용할 최대 워커 수: {max_workers if max_workers else os.cpu_count()} (시스템 CPU: {os.cpu_count()})")

    # 랜덤 시드 설정
    random.seed(seed)
    print(f"랜덤 시드 {seed}로 고정.")

    # --- 1. 입력 폴더 존재 확인 ---
    if not label_tg_path.is_dir():
        print(f"오류: 원본 TextGrid 폴더 '{label_tg_path}'를 찾을 수 없습니다.")
        return
    if not noise_mixed_audio_path.is_dir():
        print(f"오류: 원본 오디오 폴더 '{noise_mixed_audio_path}'를 찾을 수 없습니다.")
        return

    # --- 2. label_tg 폴더에서 .TextGrid 파일 목록 가져오기 ---
    print(f"'{label_tg_path}'에서 .TextGrid 파일 목록 스캔 중...")
    all_textgrid_files = list(label_tg_path.glob("*.TextGrid"))
    print(f"총 {len(all_textgrid_files)}개의 .TextGrid 파일 발견.")

    if len(all_textgrid_files) == 0:
        print(f"오류: '{label_tg_path}'에서 .TextGrid 파일을 찾을 수 없습니다.")
        return
    if len(all_textgrid_files) < n_samples:
        print(f"경고: 요청한 샘플 개수({n_samples})보다 사용 가능한 .TextGrid 파일({len(all_textgrid_files)})이 적습니다.")
        print(f"사용 가능한 모든 파일({len(all_textgrid_files)}개)을 샘플링합니다.")
        n_samples = len(all_textgrid_files)

    # --- 3. 랜덤 샘플링 수행 ---
    print(f"{len(all_textgrid_files)}개 중 {n_samples}개의 TextGrid 파일 랜덤 샘플링 중...")
    sampled_textgrid_paths = random.sample(all_textgrid_files, n_samples)
    print(f"{n_samples}개 TextGrid 파일 샘플링 완료.")

    # --- 4. 출력 폴더 생성 (메인 프로세스에서 미리 생성) ---
    sample_tg_path.mkdir(parents=True, exist_ok=True)
    sample_noise_mixed_audio_path.mkdir(parents=True, exist_ok=True)
    print(f"출력 폴더 '{sample_tg_path}' 및 '{sample_noise_mixed_audio_path}' 준비 완료.")

    # --- 5. 멀티프로세싱으로 파일 복사 실행 ---
    print(f"{max_workers if max_workers else os.cpu_count()}개의 워커 프로세스로 병렬 복사 시작...")
    copied_tg_count = 0
    copied_wav_count = 0
    not_found_wav_for_tg_count = 0
    error_count = 0
    futures = []

    # ProcessPoolExecutor를 사용하여 작업 분배
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 각 TextGrid 파일에 대해 작업 제출
        for tg_path in sampled_textgrid_paths:
            future = executor.submit(
                process_single_textgrid,
                tg_path,
                noise_mixed_audio_path,
                sample_tg_path,
                sample_noise_mixed_audio_path
            )
            futures.append(future)

        # tqdm으로 진행률 표시 및 결과 집계
        # total=n_samples 로 설정해야 정확한 진행률 표시
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_samples, desc="파일 처리 진행률"):
            try:
                tg_copied_path, num_wav_copied, wav_found_flag, error_message = future.result()

                if error_message:
                    error_count += 1
                    # 에러 메시지는 process_single_textgrid 내부 또는 여기서 출력
                    print(f"\n오류 발생 (상세): {error_message}") # 개행 문자로 tqdm 출력과 겹치는 것 방지

                if tg_copied_path: # TextGrid가 성공적으로 복사되었다면
                    copied_tg_count += 1
                    copied_wav_count += num_wav_copied # 해당 TG에 대해 복사된 WAV 수 누적
                    if not wav_found_flag: # TG는 복사했지만 WAV는 못 찾은 경우
                        not_found_wav_for_tg_count += 1
                        # 필요시 경고 메시지 추가 (어떤 파일인지 tg_copied_path.name으로 확인 가능)
                        # print(f"\n경고: TextGrid '{tg_copied_path.name}'는 복사되었으나 해당하는 WAV 파일을 찾지 못했습니다.")

                # else: TextGrid 복사 자체가 실패한 경우 (오류 메시지로 이미 처리됨)

            except Exception as e:
                # future.result()에서 예외 발생 시 (예: worker 프로세스 자체의 문제)
                error_count += 1
                print(f"\n심각한 오류: 작업 처리 중 예기치 않은 오류 발생 - {e}")

    # --- 6. 결과 요약 ---
    end_time = time.time()
    print("\n--- 작업 완료 ---")
    print(f"총 실행 시간: {end_time - start_time:.2f} 초")
    print(f"총 샘플링된 TextGrid 파일 수: {n_samples}")
    print(f"성공적으로 복사된 TextGrid 파일 수: {copied_tg_count}")
    print(f"복사된 총 오디오(.wav) 파일 수: {copied_wav_count}")
    if not_found_wav_for_tg_count > 0:
        print(f"경고: 해당하는 오디오 파일을 하나도 찾지 못한 TextGrid 파일 수: {not_found_wav_for_tg_count}")
    if error_count > 0:
        print(f"오류: 처리 중 {error_count}개의 오류 발생 (로그 확인 필요).")
    print(f"샘플 TextGrid 저장 위치: '{sample_tg_path}'")
    print(f"샘플 오디오 저장 위치: '{sample_noise_mixed_audio_path}'")
    print("-----------------")


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 메인 가드 안에서 실행해야 multiprocessing이 제대로 동작합니다 (특히 Windows).
    create_sample_dataset_parallel(
        label_tg_path=label_tg_dir,
        noise_mixed_audio_path=noise_mixed_audio_dir,
        sample_tg_path=sample_tg_dir,
        sample_noise_mixed_audio_path=sample_noise_mixed_audio_dir,
        n_samples=num_samples,
        seed=random_seed,
        max_workers=num_workers # 설정된 워커 수 전달
    )
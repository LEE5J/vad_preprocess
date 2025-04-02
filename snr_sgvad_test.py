import os
import glob
import random
import numpy as np
import librosa
import time
from tqdm import tqdm
import traceback
from sgvad import SGVAD
from parse_textgrid import parse_textgrid_to_labels
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

LABEL_TG_DIR = "./label_tg"  # TextGrid 파일이 있는 디렉토리
NOISE_MIXED_AUDIO_DIR = "./noise_mixed_audio"  # 노이즈 섞인 오디오 파일 루트 디렉토리
SGVAD_CFG_PATH = "./cfg.yaml" # SGVAD 설정 파일 경로
MAX_FILES_PER_FOLDER = 3000  # 각 하위 폴더에서 사용할 최대 파일 수
RANDOM_SEED = 42  # 파일 샘플링을 위한 랜덤 시드 고정

# VAD 결과 처리 파라미터
VAD_THRESHOLD_LOW = 0.08  # 발화 후보군 선정을 위한 낮은 임계값
VAD_THRESHOLD_HIGH = 0.15 # 실제 발화 구간 확정을 위한 높은 임계값
EVALUATION_FRAME_SHIFT_SEC = 0.01


def process_vad_scores_to_labels(scores, threshold_low, threshold_high,
                                 model_frame_shift_sec, total_duration,
                                 eval_frame_shift_sec=EVALUATION_FRAME_SHIFT_SEC):
    """
    SGVAD 모델의 프레임별 점수 리스트를 처리하여 예측된 발화 구간을 찾고,
    이를 기반으로 평가 기준 프레임 시프트 단위의 예측 레이블 배열을 반환합니다.

    Args:
        scores (list[float]): 프레임별 VAD 점수 리스트
        threshold_low (float): 낮은 임계값
        threshold_high (float): 높은 임계값
        model_frame_shift_sec (float): VAD 점수가 생성된 모델의 프레임 시프트 (초)
        total_duration (float): 오디오 전체 길이 (초)
        eval_frame_shift_sec (float): 레이블 생성을 위한 평가 기준 프레임 시프트 (초)

    Returns:
        np.ndarray | None: 예측된 프레임별 레이블 배열 (0 또는 1). 오류 시 None.
    """
    if not scores or total_duration <= 0 or eval_frame_shift_sec <= 0:
        # print(f"경고: VAD 점수 처리 입력 부족. Scores: {len(scores)>0}, Duration: {total_duration}, Eval Shift: {eval_frame_shift_sec}")
        return None # 유효하지 않은 입력

    # --- 1단계: VAD 점수를 시간 세그먼트로 변환 (모델의 프레임 시프트 기준) ---
    confirmed_speech_segments = []
    num_model_frames = len(scores)
    if num_model_frames > 0:
        candidate_frames = [i for i, score in enumerate(scores) if score >= threshold_low]
        if candidate_frames:
            potential_segments_frames = []
            start_frame = candidate_frames[0]
            current_frame = start_frame
            for i in range(1, len(candidate_frames)):
                if candidate_frames[i] == current_frame + 1:
                    current_frame = candidate_frames[i]
                else:
                    potential_segments_frames.append((start_frame, current_frame))
                    start_frame = candidate_frames[i]
                    current_frame = start_frame
            potential_segments_frames.append((start_frame, current_frame))

            for start_frame, end_frame in potential_segments_frames:
                has_high_score = any(scores[i] >= threshold_high for i in range(start_frame, end_frame + 1) if i < num_model_frames) # 범위 체크 추가
                if has_high_score:
                    # 모델 프레임 인덱스를 시간으로 변환
                    start_time = start_frame * model_frame_shift_sec
                    end_time = (end_frame + 1) * model_frame_shift_sec
                    confirmed_speech_segments.append((start_time, end_time))

    # --- 2단계: 시간 세그먼트를 평가 기준 프레임 레이블로 변환 ---
    num_eval_frames = int(np.ceil(total_duration / eval_frame_shift_sec))
    if num_eval_frames == 0:
        return np.array([], dtype=int) # 빈 레이블 배열 반환

    pred_labels = np.zeros(num_eval_frames, dtype=int) # 0으로 초기화

    for start, end in confirmed_speech_segments:
        # 시간 세그먼트를 평가 프레임 인덱스로 변환
        start_eval_frame = int(np.floor(start / eval_frame_shift_sec))
        end_eval_frame = int(np.ceil(end / eval_frame_shift_sec))
        start_eval_frame = max(0, start_eval_frame)
        end_eval_frame = min(num_eval_frames, end_eval_frame)
        if start_eval_frame < end_eval_frame:
            pred_labels[start_eval_frame:end_eval_frame] = 1

    return pred_labels



def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. SGVAD 모델 초기화
    print("SGVAD 모델 초기화 중...")
    sgvad_model = None
    sgvad_native_frame_shift_sec = 0.01 # 기본값, cfg 파일에서 읽어옴
    try:
        sgvad_model = SGVAD(cfg_path=SGVAD_CFG_PATH)
        # 모델의 네이티브 프레임 시프트 확인 (VAD 점수 처리에 사용)
        if hasattr(sgvad_model.cfg, 'preprocessor') and hasattr(sgvad_model.cfg.preprocessor, 'window_stride'):
            sgvad_native_frame_shift_sec = sgvad_model.cfg.preprocessor.window_stride
            print(f"SGVAD 모델의 네이티브 프레임 시프트: {sgvad_native_frame_shift_sec:.4f} 초")
        else:
             print(f"경고: SGVAD 설정에서 네이티브 프레임 시프트(window_stride)를 찾을 수 없음. 가정값 {sgvad_native_frame_shift_sec:.4f} 초 사용.")
        # 모델의 샘플링 레이트 확인 (오디오 로딩 시 사용될 수 있음)
        if hasattr(sgvad_model.cfg, 'sample_rate'):
             print(f"SGVAD 모델 샘플링 레이트: {sgvad_model.cfg.sample_rate} Hz")
        else:
             print("경고: SGVAD 설정에서 sample_rate를 찾을 수 없음.")

    except FileNotFoundError as e:
        print(f"오류: SGVAD 초기화 실패 - 파일({e.filename})을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"오류: SGVAD 모델 초기화 실패: {e}")
        traceback.print_exc()
        return

    # 2. 테스트할 오디오 파일 찾기 및 샘플링
    print("테스트 대상 오디오 파일 검색 및 샘플링 중...")
    all_audio_files = []
    subfolder_files = {} # 폴더명 -> 파일 경로 리스트 맵
    try:
        for subfolder_path in glob.glob(os.path.join(NOISE_MIXED_AUDIO_DIR, '*/')):
            if not os.path.isdir(subfolder_path): continue # 디렉토리가 아니면 건너<0xEB><0x9B><0x84>
            subfolder_name = os.path.basename(os.path.normpath(subfolder_path))
            wav_files = glob.glob(os.path.join(subfolder_path, '*.wav'))
            if not wav_files:
                # print(f"   정보: '{subfolder_name}' 폴더에 .wav 파일 없음.") # 로그 줄이기 위해 주석 처리
                continue

            # 파일 샘플링
            num_files_to_sample = min(len(wav_files), MAX_FILES_PER_FOLDER)
            sampled_files = random.sample(wav_files, num_files_to_sample)

            subfolder_files[subfolder_name] = sampled_files
            all_audio_files.extend(sampled_files)
            # print(f"   '{subfolder_name}' 폴더에서 {len(sampled_files)}개 파일 선택.") # 로그 줄이기 위해 주석 처리
    except Exception as e:
        print(f"오류: 오디오 파일 검색 중 오류 발생: {e}")
        return

    if not all_audio_files:
        print("오류: 테스트할 오디오 파일을 찾을 수 없습니다.")
        return

    print(f"\n총 {len(all_audio_files)}개의 오디오 파일 처리 시작 (최대 {MAX_FILES_PER_FOLDER}개/폴더)")

    # 3. SGVAD 예측 실행
    start_time = time.time()
    vad_results = [] # 각 파일에 대한 VAD 점수 리스트를 담을 리스트
    try:
        # SGVAD.predict는 파일 경로 리스트를 받아 결과 리스트 반환 가정
        print(f"{len(all_audio_files)}개 파일에 대한 VAD 예측 수행 중...")
        vad_results = sgvad_model.predict(all_audio_files) # smooth 기본값 사용
        if vad_results is None: # predict 함수가 실패를 None으로 알리는 경우
             vad_results = [] # 이후 처리를 위해 빈 리스트로
             print("경고: SGVAD predict 호출이 None을 반환했습니다.")
    except Exception as e:
        print(f"오류: SGVAD 예측 중 예외 발생: {e}")
        traceback.print_exc()
        vad_results = [] # 예측 실패 시 빈 리스트

    end_time = time.time()
    print(f"SGVAD 예측 완료 (소요 시간: {end_time - start_time:.2f} 초)")

    # 예측 결과 수 확인
    num_predictions = len(vad_results)
    num_expected = len(all_audio_files)
    if num_predictions != num_expected:
        print(f"경고: 예측 결과 수({num_predictions})가 예상({num_expected})과 다릅니다. 일부 파일 처리 실패 가능성.")
        # 길이가 다를 경우 zip 사용 시 짧은 쪽에 맞춰짐

    # 4. 파일별 처리: 라벨 생성 및 저장
    print("\n파일별 라벨 생성 및 평가 준비 중...")
    # 폴더별로 실제 레이블과 예측 레이블 리스트를 저장
    folder_all_y_true = {name: [] for name in subfolder_files}
    folder_all_y_pred = {name: [] for name in subfolder_files}
    processed_files_count_per_folder = {name: 0 for name in subfolder_files}

    # zip은 예측 결과가 부족할 경우 자동으로 짧은 길이에 맞춰 순회
    for audio_path, vad_scores in tqdm(zip(all_audio_files, vad_results), total=num_predictions, desc="파일 처리 및 라벨 생성"):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        textgrid_path = os.path.join(LABEL_TG_DIR, base_name + ".TextGrid")
        subfolder_name = os.path.basename(os.path.dirname(audio_path))

        # Ground Truth 레이블 생성
        gt_labels, total_duration_tg = parse_textgrid_to_labels(textgrid_path, EVALUATION_FRAME_SHIFT_SEC)

        # 오디오 전체 길이 결정 (TextGrid 우선, 실패 시 오디오 파일)
        total_duration = total_duration_tg
        if total_duration <= 0:
             try:
                 # 오디오 파일에서 duration 읽기 (모델의 SR 사용 시도)
                 sr = sgvad_model.cfg.sample_rate if hasattr(sgvad_model.cfg, 'sample_rate') else 16000
                 total_duration = librosa.get_duration(filename=audio_path, sr=sr)
                 if total_duration <= 0:
                     # print(f"경고: {base_name} 유효 duration 얻기 실패. 건너<0xEB><0x9B><0x84>.") # 로그 너무 많음
                     continue # 유효 duration 없으면 이 파일 처리 불가
             except Exception as e:
                 # print(f"경고: {base_name} duration 로드 실패({e}). 건너<0xEB><0x9B><0x84>.") # 로그 너무 많음
                 continue # 오디오 로드 실패 시 처리 불가

        # GT 레이블 유효성 재확인 및 재생성 시도 (duration 이 변경되었을 수 있으므로)
        expected_gt_len = int(np.ceil(total_duration / EVALUATION_FRAME_SHIFT_SEC))
        if gt_labels is None or len(gt_labels) != expected_gt_len:
             # TextGrid 파싱 실패 또는 duration 불일치 시, duration 기준으로 빈 레이블이라도 생성 시도
             # (단, TextGrid 원본 정보 없이는 발화 구간 알 수 없어 모두 0으로 처리될 수 있음)
             # -> 여기서는 GT 파싱 실패 시 그냥 건너뛰도록 함 (gt_labels is None)
             if gt_labels is None:
                  # print(f"정보: {base_name} GT 레이블 생성 불가. 건너<0xEB><0x9B><0x84>니다.")
                  continue # GT 없으면 평가 불가

        # VAD 예측 레이블 생성
        pred_labels = process_vad_scores_to_labels(vad_scores, VAD_THRESHOLD_LOW, VAD_THRESHOLD_HIGH,
                                                   sgvad_native_frame_shift_sec, total_duration,
                                                   EVALUATION_FRAME_SHIFT_SEC)

        # 예측 레이블 유효성 검사
        if pred_labels is None or len(gt_labels) != len(pred_labels):
            # print(f"경고: {base_name} 예측 레이블 생성 실패 또는 길이 불일치. 건너<0xEB><0x9B><0x84>.")
            continue

        # 해당 폴더의 결과 리스트에 추가 (길이가 0 이상일 때만)
        if subfolder_name in folder_all_y_true and len(gt_labels) > 0:
            folder_all_y_true[subfolder_name].append(gt_labels)
            folder_all_y_pred[subfolder_name].append(pred_labels)
            processed_files_count_per_folder[subfolder_name] += 1
        elif subfolder_name not in folder_all_y_true:
             print(f"경고: 파일 {audio_path}의 폴더명 '{subfolder_name}'을 찾을 수 없습니다.")


    # 5. 폴더별 성능 평가 및 결과 출력 (Classification Report 만 사용)
    print("\n" + "=" * 30)
    print(f"폴더별 성능 평가 결과 ({EVALUATION_FRAME_SHIFT_SEC*1000:.1f} ms 프레임 기준)")
    print("=" * 30)

    for subfolder_name in sorted(subfolder_files.keys()): # 폴더 이름 순서대로 출력
        print(f"\n--- 폴더: {subfolder_name} ---") # ★ 폴더 이름 출력

        processed_count = processed_files_count_per_folder.get(subfolder_name, 0)
        y_true_list = folder_all_y_true.get(subfolder_name, [])
        y_pred_list = folder_all_y_pred.get(subfolder_name, [])

        if processed_count == 0 or not y_true_list:
            print("  처리된 유효 파일이 없어 성능을 계산할 수 없습니다.")
            continue

        # 폴더 내 모든 파일의 레이블 배열을 하나로 합침
        try:
            folder_y_true = np.concatenate(y_true_list)
            folder_y_pred = np.concatenate(y_pred_list)
        except ValueError as e:
            print(f"  오류: 레이블 배열 합치기 실패 ({e}). 건너<0xEB><0x9B><0x84>니다.")
            # 각 파일의 레이블 길이나 타입이 다른 경우 발생 가능
            continue

        # --- Sklearn Classification Report 출력 ---
        try:
            # classification_report 함수 호출하여 문자열 결과 생성
            report_str = classification_report(
                folder_y_true,
                folder_y_pred,
                target_names=['Silence (0)', 'Speech (1)'], # 클래스 이름 설정
                zero_division=0, # 0으로 나누기 경고 방지 및 결과 0으로 표시
                digits=4 # 소수점 이하 자릿수
            )
            print(report_str) # ★ Classification Report 결과만 출력
        except Exception as e:
            # Classification Report 생성 중 오류 발생 시 메시지 출력
            print(f"  Classification Report 생성 중 오류 발생: {e}")
            # 디버깅 위한 추가 정보 출력 (선택적)
            try:
                unique_true, counts_true = np.unique(folder_y_true, return_counts=True)
                unique_pred, counts_pred = np.unique(folder_y_pred, return_counts=True)
                print(f"    실제 레이블 분포: {dict(zip(unique_true, counts_true))}")
                print(f"    예측 레이블 분포: {dict(zip(unique_pred, counts_pred))}")
            except Exception as ie:
                 print(f"    레이블 분포 확인 중 추가 오류: {ie}")



    print("\n" + "=" * 30)
    print("테스트 완료.")


if __name__ == "__main__":
    # 이 스크립트가 직접 실행될 때 main 함수 호출
    main()

import traceback
import numpy as np


def parse_textgrid_to_labels(textgrid_path, frame_shift_sec=0.01):
    """
    TextGrid 파일을 파싱하여 발화 구간을 찾고, 이를 기반으로
    지정된 프레임 시프트 단위의 Ground Truth 레이블 배열을 반환합니다.

    Args:
        textgrid_path (str): TextGrid 파일 경로
        frame_shift_sec (float): 레이블 생성을 위한 프레임 이동 시간 (초)

    Returns:
        tuple: (gt_labels, total_duration)
            gt_labels (np.ndarray | None): 프레임별 레이블 배열 (0 또는 1). 오류 시 None.
            total_duration (float): TextGrid 파일의 전체 길이 (xmax). 오류 시 0.0.
    """
    speech_segments = []
    total_duration = 0.0
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # --- TextGrid 파싱 로직 (기존과 유사) ---
        in_word_tier = False
        intervals_size = 0
        intervals_processed = 0
        object_xmax_found = False

        for i, line in enumerate(lines):
            line = line.strip()
            if not object_xmax_found and line.startswith("xmax ="):
                 is_global_xmax = True
                 for j in range(i):
                     if lines[j].strip().startswith("item ["):
                         is_global_xmax = False
                         break
                 if is_global_xmax:
                    try:
                        total_duration = float(line.split("=")[1].strip())
                        object_xmax_found = True
                    except ValueError: pass

            if 'name = "words"' in line: in_word_tier = True
            elif in_word_tier and "intervals: size =" in line:
                try: intervals_size = int(line.split("=")[1].strip())
                except ValueError: in_word_tier = False
            elif in_word_tier and "intervals [" in line: xmin, xmax, text = None, None, None
            elif in_word_tier and "xmin =" in line:
                try: xmin = float(line.split("=")[1].strip())
                except ValueError: xmin = None
            elif in_word_tier and "xmax =" in line:
                try: xmax = float(line.split("=")[1].strip())
                except ValueError: xmax = None
            elif in_word_tier and "text =" in line:
                text = line.split("=")[1].strip().strip('"')
                if xmin is not None and xmax is not None and text is not None:
                    if text != "": speech_segments.append((xmin, xmax))
                    intervals_processed += 1
                    xmin, xmax, text = None, None, None
                    if intervals_size > 0 and intervals_processed == intervals_size:
                        in_word_tier = False

        # Fallback duration logic (기존과 동일)
        if total_duration == 0.0 and speech_segments:
             try:
                 last_interval_end = speech_segments[-1][1]
                 total_duration = last_interval_end
                 # print(f"경고: {textgrid_path} global xmax 못찾음. 마지막 발화 끝({total_duration:.3f}s) 사용.")
             except Exception: pass

        # --- 파싱된 세그먼트를 프레임 레이블로 변환 ---
        if total_duration <= 0 or frame_shift_sec <= 0:
            # print(f"경고: {textgrid_path} 유효한 duration({total_duration}) 또는 frame_shift({frame_shift_sec}) 없음.")
            # Duration이 0이면 오디오 파일에서 다시 시도해야 함
            return None, total_duration # 레이블 생성 불가, duration은 반환

        num_frames = int(np.ceil(total_duration / frame_shift_sec))
        if num_frames == 0:
            return np.array([], dtype=int), total_duration # 빈 레이블 배열 반환

        gt_labels = np.zeros(num_frames, dtype=int) # bool 대신 int 사용 (sklearn 호환성)

        for start, end in speech_segments:
            start_frame = int(np.floor(start / frame_shift_sec))
            end_frame = int(np.ceil(end / frame_shift_sec))
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)
            if start_frame < end_frame:
                 gt_labels[start_frame:end_frame] = 1

        return gt_labels, total_duration

    except FileNotFoundError:
        print(f"오류: TextGrid 파일을 찾을 수 없습니다: {textgrid_path}")
        return None, 0.0
    except Exception as e:
        print(f"오류: TextGrid 파일 파싱 중 오류 발생 ({textgrid_path}): {e}")
        traceback.print_exc()
        return None, 0.0

if __name__ == "__main__":
    # 테스트용 코드
    test_path = "label_tg/sample_1.TextGrid"
    segments, duration = parse_textgrid_to_labels(test_path)
    print(f"발화 구간: {segments}")
    print(f"전체 길이: {duration}")
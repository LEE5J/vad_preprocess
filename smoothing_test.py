# smoothing_test.py
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm # Add tqdm for progress indication

# Import the SGVAD class from sgvad.py
try:
    from sgvad import SGVAD
except ImportError:
    print("오류: sgvad.py 파일을 찾거나 임포트할 수 없습니다.")
    print("smoothing_test.py와 같은 디렉토리에 sgvad.py가 있는지 확인하세요.")
    exit(1)

# --- 결과 저장 디렉토리 생성 ---
PLOT_OUTPUT_DIR = "vad_plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
print(f"VAD 플롯은 '{PLOT_OUTPUT_DIR}' 디렉토리에 저장됩니다.")

def plot_vad_scores(scores, fpath_out=None, title="VAD Analysis",
                    figsize=(15, 5), dpi=100, smoothing_window_vis=15):
    """
    VAD 결과 시각화 함수 (sgvad.py에서 얻은 스무딩된 점수 사용)

    Parameters:
        scores (np.ndarray): sgvad.predict()에서 반환된 스무딩된 프레임 점수 배열.
        fpath_out (str): 그래프 저장 경로 (None시 화면 표시).
        title (str): 그래프 제목.
        figsize (tuple): 그래프 크기.
        dpi (int): 이미지 해상도.
        smoothing_window_vis (int): 시각화용 이동 평균 창 크기 (필요시 추가 스무딩).
                                    sgvad.py의 스무딩과 별개일 수 있음.
    """
    if scores is None or len(scores) == 0:
        print(f"경고: 점수가 없거나 비어 있어 플롯을 생성할 수 없습니다. ({title})")
        return

    scores = np.squeeze(np.array(scores)) # Ensure numpy array and remove single dims

    plt.figure(figsize=figsize, dpi=dpi)
    # 프레임 인덱스 또는 추정 시간 축 생성 (sgvad.py 내부 프레임 설정에 따라 조정 필요 가능성 있음)
    # 예시: NeMo 기본값 hop_length=0.01초 가정
    frame_duration_sec = 0.01 # 이 값은 sgvad.py의 preprocessor 설정과 일치해야 함
    time_axis = np.arange(len(scores)) * frame_duration_sec

    # sgvad.py에서 이미 스무딩된 점수를 직접 플롯
    plt.plot(time_axis, scores,
             color='red', linewidth=1.5,
             label=f'Smoothed Scores (from sgvad.py)')

    # 시각화를 위한 추가 스무딩 (선택 사항)
    if smoothing_window_vis > 1 and len(scores) >= smoothing_window_vis:
        kernel_vis = np.ones(smoothing_window_vis) / smoothing_window_vis
        vis_smoothed_scores = np.convolve(scores, kernel_vis, mode='same')
        plt.plot(time_axis, vis_smoothed_scores,
                 color='purple', linewidth=1.0, linestyle=':',
                 label=f'Visually Smoothed (Window={smoothing_window_vis})')


    # 시각적 임계값 (예: 백분위수) 추가
    if len(scores) > 0:
        t_A = 70
        t_B = 95
        try:
            threshold_A = np.percentile(scores, t_A)
            threshold_B = np.percentile(scores, t_B)
            plt.axhline(threshold_A, color='green',
                        linestyle='--', label=f'{t_A}th percentile ({threshold_A:.2f})')
            plt.axhline(threshold_B, color='orange', # 색상 변경 (노란색은 잘 안 보일 수 있음)
                        linestyle='--', label=f'{t_B}th percentile ({threshold_B:.2f})')

            # 임계값 기준으로 영역 강조 (예: threshold_B 사용)
            plt.fill_between(time_axis, 0, 1, # y축 범위를 0과 1 사이로 고정
                             where=(scores > threshold_B),
                             color='red', alpha=0.2, label=f'Above {t_B}th percentile')

        except IndexError:
             print(f"경고: 백분위수 계산 중 오류 발생 ({title}). 점프합니다.")


    plt.xlabel('Time (seconds, estimated)')
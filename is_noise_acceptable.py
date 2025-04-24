import librosa
import numpy as np

def score_wav_file(file_path):
    # WAV 파일을 16kHz 샘플레이트로 로드
    y, sr = librosa.load(file_path, sr=16000)
    rms = np.sqrt(np.mean(y**2))
    # print(f"RMS: {rms}")
    
    # 절대값 변환 (진폭의 크기만 고려)
    y_abs = np.abs(y)
    
    # 95 퍼센타일 계산
    percentile_95 = np.percentile(y_abs, 95)
    # print(f"95 퍼센타일: {percentile_95}")
    
    # 5 퍼센타일 계산
    percentile_5 = np.percentile(y_abs, 15)
    # print(f"5 퍼센타일: {percentile_5}")
    
    # 비율 계산 (0으로 나누는 것을 방지)
    if percentile_5 > 0:
        ratio = percentile_95 / percentile_5
    else:
        # 5 퍼센타일이 0인 경우 (매우 조용한 부분이 있는 경우)
        # 작은 값으로 대체하여 높은 비율이 나오도록 함
        ratio = percentile_95 / 1e-10
    
    
    score = ratio*4 - 60
    score = max(0, min(100, score))  # 0~100 사이로 클리핑
    
    return score

if __name__ == "__main__":
    for i in range(1, 5):
        try:
            score = score_wav_file(f'sample_{i}.wav')
            print(f"파일 점수: {score:.2f}")
            print("-" * 30)
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {e}")

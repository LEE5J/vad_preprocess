import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# 함수 정의: 하위 폴더 각각에서 최대 1000개의 wav 파일을 읽고 RMS로 정규화 후 20개 퍼센타일 계산
def calculate_percentiles_by_subfolder(parent_folder):
    # 하위 폴더 목록 가져오기
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    folder_percentiles = {}
    
    # 각 하위 폴더별로 처리
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"처리 중인 폴더: {folder_name}")
        
        # 해당 폴더의 wav 파일 최대 1000개 가져오기
        wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')][:1000]
        
        if not wav_files:
            print(f"경고: {folder_name} 폴더에 wav 파일이 없습니다.")
            continue
            
        print(f"파일 수: {len(wav_files)}")
        
        all_normalized_rms = []
        
        # 각 wav 파일 처리
        for wav_file in wav_files:
            file_path = os.path.join(folder, wav_file)
            try:
                # 16kHz로 음원 로드
                y, sr = librosa.load(file_path, sr=16000)
                # RMS 계산
                rms = np.sqrt(np.mean(y**2))
                # RMS로 정규화
                normalized = y / rms if rms > 0 else y
                # 정규화된 신호의 절대값 저장
                all_normalized_rms.extend(np.abs(normalized))
            except Exception as e:
                print(f"파일 {wav_file} 처리 중 오류 발생: {e}")
                continue
        
        if not all_normalized_rms:
            print(f"경고: {folder_name} 폴더에서 처리 가능한 파일이 없습니다.")
            continue
            
        all_normalized_rms = np.array(all_normalized_rms)
        
        # 20개 퍼센타일 값 계산 (90%에서 99.9%까지)
        percentile_values = np.linspace(90, 99.9, 100)
        percentiles = np.percentile(all_normalized_rms, percentile_values)
        
        # 결과 저장
        folder_percentiles[folder_name] = percentiles
        
        # 결과 출력 (디버깅용)
        print(f"{folder_name} 퍼센타일 계산 완료")
    
    return folder_percentiles

# 그래프 그리기 함수
def plot_percentiles(folder_percentiles):
    plt.figure(figsize=(12, 8))
    
    # x축 값 (0부터 19까지)
    x_indices = np.arange(100)
    all_percentiles = []
    # 각 폴더별 퍼센타일 그래프 그리기
    for folder_name, percentiles in folder_percentiles.items():
        plt.plot(x_indices, percentiles, label=folder_name, linewidth=2, marker='o', markersize=4)
        all_percentiles.append(percentiles)
    
    diff = []
    for i in range(len(all_percentiles[0])):
        diff.append(np.max(all_percentiles, axis=0)[i] - np.min(all_percentiles, axis=0)[i])
    print(np.max(diff), diff.index(np.max(diff)))

    arr   = np.array(all_percentiles)            # shape: (6, 100)
    diff  = arr.max(axis=0) - arr.min(axis=0)    # shape: (100,)
    max_d = diff.max()                           # 최댓값
    idx   = diff.argmax()                        # 최댓값 인덱스
    print(max_d, idx)
    
    
    plt.xlabel('퍼센타일 인덱스 (0-19)', fontsize=12)
    plt.ylabel('RMS 정규화 퍼센타일 값', fontsize=12)
    plt.title('폴더별 RMS 정규화 퍼센타일 비교', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig('noisepercentile.png', dpi=300)
    print("그래프가 'noisepercentile.png'로 저장되었습니다.")
    plt.close()

# 메인 함수
def main():
    # 폴더 경로 입력 (실제 사용시 변경 필요)
    parent_folder = "noise_mixed_audio2"
    
    if not os.path.exists(parent_folder):
        print(f"오류: 경로 '{parent_folder}'가 존재하지 않습니다.")
        return
    
    # 퍼센타일 계산
    folder_percentiles = calculate_percentiles_by_subfolder(parent_folder)
    
    if not folder_percentiles:
        print("처리할 데이터가 없습니다.")
        return
    
    # 그래프 그리기
    plot_percentiles(folder_percentiles)

# 프로그램 실행
if __name__ == "__main__":
    main()
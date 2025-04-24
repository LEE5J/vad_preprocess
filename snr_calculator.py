import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def calculate_percentiles_by_subfolder(parent_folder):
    # 하위 폴더 목록 가져오기
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    results = {}
    
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
        
        # 5퍼센타일과 95퍼센타일 계산
        percentile_5 = np.percentile(all_normalized_rms, 15)
        percentile_95 = np.percentile(all_normalized_rms, 95)
        
        # 결과 저장
        results[folder_name] = {
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'ratio': percentile_95 / percentile_5 if percentile_5 > 0 else float('inf')
        }
        
        # 결과 출력
        print(f"{folder_name}:")
        print(f"  5 퍼센타일: {percentile_5:.6f}")
        print(f" 95 퍼센타일: {percentile_95:.6f}")
        print(f" 95/5 비율: {results[folder_name]['ratio']:.6f}")
        print("-" * 40)
    
    return results

def plot_percentile_ratios(results):
    folder_names = list(results.keys())
    ratios = [results[folder]['ratio'] for folder in folder_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(folder_names, ratios)
    plt.xlabel('폴더명')
    plt.ylabel('95퍼센타일 / 5퍼센타일 비율')
    plt.title('폴더별 95/5 퍼센타일 비율')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig('percentile_ratio.png', dpi=300)
    print("그래프가 'percentile_ratio.png'로 저장되었습니다.")
    plt.close()

def main():
    # 폴더 경로 입력 (실제 사용시 변경 필요)
    parent_folder = "noise_mixed_audio2"
    
    if not os.path.exists(parent_folder):
        print(f"오류: 경로 '{parent_folder}'가 존재하지 않습니다.")
        return
    
    # 퍼센타일 계산
    results = calculate_percentiles_by_subfolder(parent_folder)
    
    if not results:
        print("처리할 데이터가 없습니다.")
        return
    
    # 요약 정보 출력
    print("\n=== 폴더별 95/5 퍼센타일 비율 요약 ===")
    for folder_name, data in results.items():
        print(f"{folder_name}: {data['ratio']:.6f}")
    
    # 그래프 그리기
    plot_percentile_ratios(results)

# 프로그램 실행
if __name__ == "__main__":
    main()

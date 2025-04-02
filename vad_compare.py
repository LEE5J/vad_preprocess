import os
import glob
import random
import numpy as np
import textgrid
from pydub import AudioSegment
from tqdm import tqdm
import torch
import whisper
import traceback
from typing import List, Dict, Tuple, Any

from vad_compare_util import ErrorCollector
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"




# Whisper 기반 VAD 모델 (배치 처리 지원)
class WhisperVAD:
    def __init__(self, model_name="large-v2", device=None, batch_size=8, error_collector=None):
        if device is None:
            device = f"cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model {model_name} on {device}")
        self.model = whisper.load_model(model_name, device=device)
        self.device = device
        self.batch_size = batch_size
        self.error_collector = error_collector or ErrorCollector()
        
    def detect_speech_batch(self, audio_paths):
        """배치로 여러 오디오 파일의 발화 구간을 감지합니다."""
        results = {}
        
        # 배치 크기만큼 파일들을 그룹화하여 처리 (tqdm 추가)
        for i in range(0, len(audio_paths), self.batch_size):
            batch_paths = audio_paths[i:i+self.batch_size]
            
            # 각 파일별로 처리 (whisper는 파일별 처리만 지원)
            for audio_path in batch_paths:
                try:
                    # 오디오 파일 검사
                    if not os.path.exists(audio_path):
                        self.error_collector.add("File Access", f"File does not exist: {audio_path}")
                        results[audio_path] = []
                        continue
                    
                    # 오디오 파일 정보 출력 (디버깅용)
                    try:
                        audio = AudioSegment.from_wav(audio_path)
                    except Exception as audio_err:
                        self.error_collector.add("Audio Reading", f"{audio_path}: {type(audio_err).__name__}: {audio_err}")
                    
                    # Whisper 모델로 처리
                    result = self.model.transcribe(
                        audio_path,
                        word_timestamps=True,
                        beam_size=5,
                        patience=2,
                        temperature=0.1,
                        language="en"
                    )
                    
                    segments = []
                    for segment in result["segments"]:
                        for word_info in segment["words"]:
                            text = word_info["word"].strip()
                            if text:  # 공백 제외
                                start_time = word_info["start"]
                                end_time = word_info["end"]
                                
                                # 시작과 끝 시간이 같으면 제외
                                if abs(end_time - start_time) < 0.001:
                                    continue
                                
                                segments.append((start_time, end_time, text))
                    
                    results[audio_path] = segments
                    
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    traceback_str = traceback.format_exc()
                    self.error_collector.add("Speech Detection", f"{audio_path}: {error_type}: {error_msg}")
                    results[audio_path] = []
        
        return results

# 구간 병합 및 정렬 함수
def merge_overlapping_segments(segments, max_gap=0.3):
    """겹치는 발화 구간을 병합하고, 시작/끝이 같은 구간을 처리합니다."""
    if not segments:
        return []
    
    # 시작 시간으로 정렬
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    # 유효한 구간만 선택 (시작과 끝이 다른 구간)
    valid_segments = []
    for start, end, text in sorted_segments:
        if abs(end - start) > 0.001:  # 길이가 있는 구간만 선택
            valid_segments.append((start, end, text))
    
    if not valid_segments:
        return []
    
    # 겹치는 구간 병합 및 근접 구간 병합
    merged_segments = []
    current = valid_segments[0]
    
    for next_segment in valid_segments[1:]:
        current_start, current_end, current_text = current
        next_start, next_end, next_text = next_segment
        
        # 구간이 겹치거나 매우 가까운 경우 (max_gap 이내)
        if next_start <= current_end + max_gap:
            # 구간 병합
            merged_end = max(current_end, next_end)
            current = (current_start, merged_end, current_text + " " + next_text)
        else:
            # 충분히 떨어진 구간은 별도로 저장
            merged_segments.append(current)
            current = next_segment
    
    # 마지막 구간 추가
    merged_segments.append(current)
    
    return merged_segments


def create_textgrid(file_path, segments, output_path, error_collector):
    """발화 세그먼트로부터 TextGrid 파일을 생성합니다."""
    try:
        # 오디오 파일 로딩
        audio = AudioSegment.from_wav(file_path)
        duration_seconds = len(audio) / 1000.0
        
        # 세그먼트가 없으면 빈 TextGrid 생성
        if not segments:
            error_collector.add("TextGrid Creation", f"No segments found for {file_path}, creating empty TextGrid")
        
        # 구간 병합 및 겹침 처리 (0.3초 이내 간격은 병합)
        merged_segments = merge_overlapping_segments(segments, max_gap=0.3)
        
        tg = textgrid.TextGrid()
        word_tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=duration_seconds)

        # 처음에 빈 구간 추가
        if merged_segments and merged_segments[0][0] > 0:
            word_tier.addInterval(textgrid.Interval(0.0, merged_segments[0][0], ""))
        
        # 모든 세그먼트 순회하며 구간 추가
        last_end = 0.0
        for i, (start, end, text) in enumerate(merged_segments):
            # 시간 범위 유효성 검사
            start = max(0.0, min(start, duration_seconds))
            end = max(start + 0.001, min(end, duration_seconds))
            
            # 이전 구간과의 간격에 빈 구간 추가
            if start > last_end and i > 0:
                try:
                    word_tier.addInterval(textgrid.Interval(last_end, start, ""))
                except Exception as interval_err:
                    error_collector.add("TextGrid Interval", f"Error adding empty interval: {interval_err}")
            
            # 발화 구간 추가
            try:
                word_tier.addInterval(textgrid.Interval(start, end, text))
                last_end = end
            except Exception as interval_err:
                error_collector.add("TextGrid Interval", 
                                   f"Error adding interval: {type(interval_err).__name__}: {interval_err}, "
                                   f"Skipping: start={start}, end={end}")
        
        # 마지막 빈 구간 추가
        if last_end < duration_seconds:
            try:
                word_tier.addInterval(textgrid.Interval(last_end, duration_seconds, ""))
            except Exception as interval_err:
                error_collector.add("TextGrid Interval", f"Error adding final empty interval: {interval_err}")

        tg.append(word_tier)
        
        # 부모 디렉토리 확인 및 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # TextGrid 저장
        try:
            tg.write(output_path)
        except Exception as write_err:
            error_collector.add("TextGrid Writing", f"{output_path}: {type(write_err).__name__}: {write_err}")
            
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        error_collector.add("TextGrid Creation", f"{file_path}: {error_type}: {error_msg}")
        
        # 오류가 발생해도 빈 TextGrid라도 생성
        try:
            tg = textgrid.TextGrid()
            word_tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=duration_seconds)
            word_tier.addInterval(textgrid.Interval(0.0, duration_seconds, ""))
            tg.append(word_tier)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tg.write(output_path)
            error_collector.add("TextGrid Recovery", f"Created empty fallback TextGrid for {file_path}")
        except:
            error_collector.add("TextGrid Recovery", f"Failed to create even an empty TextGrid for {file_path}")

# TextGrid 읽기 함수
def read_textgrid(tg_path, error_collector):
    """TextGrid 파일을 읽어 세그먼트 목록을 반환합니다."""
    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
        segments = []
        
        for tier in tg:
            if tier.name == "words":
                for interval in tier:
                    if interval.mark.strip():  # 비어있지 않은 텍스트만 발화로 간주
                        segments.append((interval.minTime, interval.maxTime, interval.mark))
        
        return segments
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        error_collector.add("TextGrid Reading", f"{tg_path}: {error_type}: {error_msg}")
        return []

# 타임라인 변환 함수
def convert_to_timeline(segments, duration, sample_rate=100, error_collector=None):
    """세그먼트 목록을 타임라인 배열로 변환합니다."""
    try:
        # 적절한 타임라인 길이 확보
        if duration <= 0:
            if error_collector:
                error_collector.add("Timeline Conversion", f"Invalid duration {duration}, using default value")
            duration = 1.0  # 기본값 설정
            
        # 충분한 길이의 배열 생성
        timeline_length = int(duration * sample_rate) + 1
        timeline = np.zeros(timeline_length, dtype=np.int8)
        
        for start, end, _ in segments:
            # 값 범위 검사
            if start < 0 or end < 0:
                if error_collector:
                    error_collector.add("Timeline Conversion", f"Negative time values: start={start}, end={end}")
                continue
                
            start_idx = max(0, int(start * sample_rate))
            end_idx = min(int(end * sample_rate), timeline_length - 1)
            
            if start_idx <= end_idx:
                timeline[start_idx:end_idx+1] = 1
            else:
                if error_collector:
                    error_collector.add("Timeline Conversion", f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}")
        
        return timeline
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        if error_collector:
            error_collector.add("Timeline Conversion", f"{error_type}: {error_msg}")
        # 오류 시 빈 타임라인 반환
        return np.zeros(int(max(duration, 1.0) * sample_rate) + 1, dtype=np.int8)

# 평가 지표 계산 함수
def calculate_metrics(pred_timeline, true_timeline, error_collector=None):
    """예측 타임라인과 정답 타임라인을 비교하여 평가 지표를 계산합니다."""
    try:
        # 길이 검사 및 조정
        if len(pred_timeline) != len(true_timeline):
            if error_collector:
                error_collector.add("Metrics Calculation", 
                                   f"Timeline length mismatch: pred={len(pred_timeline)}, true={len(true_timeline)}")
            min_length = min(len(pred_timeline), len(true_timeline))
            pred_timeline = pred_timeline[:min_length]
            true_timeline = true_timeline[:min_length]
            
        tp = np.sum((pred_timeline == 1) & (true_timeline == 1))
        fp = np.sum((pred_timeline == 1) & (true_timeline == 0))
        fn = np.sum((pred_timeline == 0) & (true_timeline == 1))
        tn = np.sum((pred_timeline == 0) & (true_timeline == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        if error_collector:
            error_collector.add("Metrics Calculation", f"{error_type}: {error_msg}")
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "accuracy": 0
        }

# 배치 처리 함수
def process_batch(vad_model, batch_files, batch_subfolders, output_dir, label_dir, sample_rate, results, error_collector, debug_mode=False):
    """파일 배치를 처리하고 결과를 평가합니다."""
    if debug_mode:
        print(f"\nProcessing batch of {len(batch_files)} files")
        
    # 배치로 발화 감지
    speech_segments_batch = vad_model.detect_speech_batch(batch_files)
    
    # 각 파일별 결과 처리
    for idx, file_path in enumerate(batch_files):
        try:
            subfolder = batch_subfolders[idx]
            
            # 출력 폴더 생성
            output_subfolder = os.path.join(output_dir, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)
            
            # 파일명 추출 (확장자 제외)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            if debug_mode:
                print(f"Processing result for {base_name} in {subfolder}")
            
            # 발화 구간
            speech_segments = speech_segments_batch.get(file_path, [])
            
            # TextGrid 생성 및 저장
            output_path = os.path.join(output_subfolder, f"{base_name}.TextGrid")
            create_textgrid(file_path, speech_segments, output_path, error_collector)
            
            # 정답 TextGrid 읽기
            label_path = os.path.join(label_dir, f"{base_name}.TextGrid")
            if not os.path.exists(label_path):
                error_collector.add("Label Files", f"정답 파일을 찾을 수 없습니다: {label_path}")
                continue
            
            true_segments = read_textgrid(label_path, error_collector)
            
            # 오디오 길이 가져오기
            try:
                audio = AudioSegment.from_wav(file_path)
                duration_seconds = len(audio) / 1000.0
                
                if debug_mode:
                    print(f"Audio duration: {duration_seconds}s")
                    print(f"Speech segments: {len(speech_segments)}")
                    print(f"True segments: {len(true_segments)}")
                
                # 타임라인으로 변환 (샘플레이트에 맞춰서)
                pred_timeline = convert_to_timeline(speech_segments, duration_seconds, sample_rate, error_collector)
                true_timeline = convert_to_timeline(true_segments, duration_seconds, sample_rate, error_collector)
                
                # 평가 지표 계산
                metrics = calculate_metrics(pred_timeline, true_timeline, error_collector)
                
                # 결과 저장
                for metric, value in metrics.items():
                    results[subfolder][metric].append(value)
                    results["overall"][metric].append(value)
                
                if debug_mode:
                    print(f"Metrics: {metrics}")
            except Exception as audio_err:
                error_collector.add("Audio Processing", f"{file_path}: {type(audio_err).__name__}: {audio_err}")
                if debug_mode:
                    traceback.print_exc()
                
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            error_collector.add("Batch Processing", f"{file_path}: {error_type}: {error_msg}")
            if debug_mode:
                traceback.print_exc()

# 메인 평가 함수
def evaluate_vad(
    input_dir="noise_mixed_audio", 
    output_dir="result_tg", 
    label_dir="label_tg",
    model_name="large-v2",
    sample_limit=None,
    sample_rate=100,
    batch_size=32,
    debug_mode=True
):
    """VAD 모델을 평가합니다."""
    # 에러 수집기 초기화
    error_collector = ErrorCollector()
    
    # 모델 초기화
    vad_model = WhisperVAD(model_name=model_name, batch_size=batch_size, error_collector=error_collector)
    
    # 입력 폴더의 하위 폴더 찾기
    try:
        subfolders = [f for f in os.listdir(input_dir) 
                    if os.path.isdir(os.path.join(input_dir, f))]
    except Exception as e:
        error_collector.add("Directory Access", f"Error accessing directory {input_dir}: {type(e).__name__}: {e}")
        error_collector.print_all()
        return None
    
    all_files = []
    
    # 각 하위 폴더에서 wav 파일 찾기
    for subfolder in subfolders:
        try:
            subfolder_path = os.path.join(input_dir, subfolder)
            wav_files = glob.glob(os.path.join(subfolder_path, "*.wav"))
            
            if debug_mode:
                print(f"Found {len(wav_files)} files in {subfolder_path}")
                if len(wav_files) > 0:
                    print(f"Example file: {wav_files[0]}")
            
            # 각 폴더별로 샘플 제한 적용
            if sample_limit and len(wav_files) > sample_limit:
                print(f"{subfolder}: 무작위로 선택한 {sample_limit}개 파일로 제한합니다.")
                random.seed(42)  # 재현 가능성을 위한 시드 설정
                wav_files = random.sample(wav_files, sample_limit)
            
            for wav_file in wav_files:
                all_files.append((subfolder, wav_file))
        except Exception as e:
            error_collector.add("Folder Scanning", f"Error scanning folder {subfolder}: {type(e).__name__}: {e}")
    
    print(f"{len(subfolders)}개 폴더에서 총 {len(all_files)}개 오디오 파일을 처리합니다.")
    
    # 폴더 스캔 관련 에러 출력
    if error_collector.errors:
        print("\n폴더 스캔 중 발생한 에러:")
        error_collector.print_all()
        error_collector.clear()
    
    # 결과 저장용 딕셔너리
    results = {
        "overall": {"precision": [], "recall": [], "f1": [], "accuracy": []}
    }
    
    for subfolder in subfolders:
        results[subfolder] = {"precision": [], "recall": [], "f1": [], "accuracy": []}
    
    # 배치 처리를 위한 준비
    batch_files = []
    batch_subfolders = []
    
    # tqdm으로 진행 상황 표시
    with tqdm(total=len(all_files), desc="전체 파일 처리 중") as pbar:
        for subfolder, file_path in all_files:
            batch_files.append(file_path)
            batch_subfolders.append(subfolder)
            
            # 배치가 모이면 처리
            if len(batch_files) >= batch_size:
                process_batch(vad_model, batch_files, batch_subfolders, output_dir, label_dir, sample_rate, results, error_collector, debug_mode)
                pbar.update(len(batch_files))  # 진행 상황 업데이트
                batch_files = []
                batch_subfolders = []
        
        # 남은 파일들 처리
        if batch_files:
            process_batch(vad_model, batch_files, batch_subfolders, output_dir, label_dir, sample_rate, results, error_collector, debug_mode)
            pbar.update(len(batch_files))  # 진행 상황 업데이트
    
    # 파일 처리 관련 에러 출력
    if error_collector.errors:
        print("\n파일 처리 중 발생한 에러:")
        error_collector.print_all()
    
    # 결과 출력
    print("\n===== 폴더별 결과 =====")
    for subfolder in sorted(subfolders):
        if not results[subfolder]["precision"]:  # 결과가 없으면 건너뛰기
            continue
            
        print(f"\n폴더: {subfolder}")
        for metric in ["precision", "recall", "f1", "accuracy"]:
            values = results[subfolder][metric]
            avg_value = sum(values) / len(values) if values else 0
            print(f"{metric.capitalize()}: {avg_value:.4f}")
    
    print("\n===== 전체 결과 =====")
    for metric in ["precision", "recall", "f1", "accuracy"]:
        values = results["overall"][metric]
        avg_value = sum(values) / len(values) if values else 0
        print(f"{metric.capitalize()}: {avg_value:.4f}")
    
    return results

# 간단한 실행 예시
if __name__ == "__main__":
    # 기본 설정으로 실행
    evaluate_vad(
        input_dir="noise_mixed_audio",
        output_dir="result_tg",
        label_dir="label_tg",
        model_name="large-v2",
        sample_limit=1000,  # 각 폴더별로 3000개 파일로 제한
        batch_size=8,      # 배치 크기
        debug_mode=False    # 디버깅 정보 출력
    )
import os
import glob
import whisper
import textgrid
from pydub import AudioSegment
from tqdm import tqdm
import torch


def create_textgrid(file_path, result, output_dir):
    audio = AudioSegment.from_wav(file_path)
    duration_seconds = len(audio) / 1000.0

    tg = textgrid.TextGrid()
    word_tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=duration_seconds)

    pad = 0.0
    for segment in result["segments"]:
        for word_info in segment["words"]:
            text = word_info["word"].strip()
            if text:  # 공백 제외
                start_time = word_info["start"] + pad
                pad = 0.0
                end_time = word_info["end"]
                while start_time >= end_time:
                    end_time += 0.05
                    pad += 0.05
                word_tier.addInterval(
                    textgrid.Interval(start_time, end_time, text)
                )

    tg.append(word_tier)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.TextGrid")
    tg.write(output_path)
    print(f"Saved TextGrid: {output_path}")

# 작업 처리 함수 (GPU에서 실행)
def transcribe_and_save(model, file_path, output_dir):
    result = model.transcribe(
        file_path,
        word_timestamps=True,
        beam_size=5,
        patience=2,
        temperature=0.1,
        language="ko"
    )
    create_textgrid(file_path, result, output_dir)


def main():

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    # 데이터셋 경로 및 출력 경로 설정
    DIR_PATH = "target_audio"
    TARG_PATH = "result_tg"
    os.makedirs(TARG_PATH, exist_ok=True)

    # 오디오 파일 정렬 및 분할
    audio_files = sorted(glob.glob(os.path.join(DIR_PATH, "*.wav")))
    num_files = len(audio_files)
    
   
    print(f"Loading Whisper model on {device}")
    model = whisper.load_model("large-v2", device=device)

    # 파일 처리 루프
    for file_path in tqdm(audio_files):
        try:
            transcribe_and_save(model, file_path, TARG_PATH)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()

#sgvad.py
import glob
import torch
from omegaconf import OmegaConf, DictConfig
from nemo.collections.asr.modules import AudioToMFCCPreprocessor, ConvASREncoder
import librosa
import os
import numpy as np
from tqdm import tqdm # 진행률 표시줄

class SGVAD:
    def __init__(self):
        """
        SGVAD 클래스 초기화. 설정 파일과 체크포인트 파일을 로드하여 모델을 준비합니다.
        GPU 사용 가능 여부를 확인하고 설정에 따라 장치를 결정합니다.
        cfg 파일 더이상 미사용
        """
        print(f"SGVAD 초기화 시작: 코드내 설정 사용")
        self.device = None
        self.cfg = {
        "sample_rate": 16000,
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "n_mels": 32,
        "n_mfcc": 32,
        "n_fft": 512,
        }
        try:
            # --- 장치 결정 ---
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
            # 모델 로드
            script_pt_file = "sgvad_model.pt"
            if not os.path.isfile(script_pt_file):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {script_pt_file}")
            model_script = torch.jit.load(script_pt_file, map_location=self.device)
            model_script.eval()
            self.model = model_script
            # ** 주의: NeMo 모듈 초기화 시 device 인수가 없을 수 있음.
            #    먼저 CPU에서 생성 후 .to(device)로 이동하는 것이 안전함.
            self.preprocessor = AudioToMFCCPreprocessor(**self.cfg).to(self.device)
            print(f"모델 및 전처리기를 장치 '{self.device}'로 이동 중...")
            # 생성된 모듈들을 결정된 장치(GPU 또는 CPU)로 이동
            self.preprocessor.to(self.device)
            self.model.to(self.device)
            print("SGVAD 초기화 완료.")
        except FileNotFoundError as e:
            print(f"초기화 오류: 필수 파일 누락 - {e}")
            raise
        except (KeyError, ValueError) as e:
            print(f"초기화 오류: 설정 또는 체크포인트 파일 내용 문제 - {e}")
            raise
        except Exception as e:
            print(f"초기화 중 예기치 않은 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise


    def load_audio(self, fpath):
        """
        지정된 경로에서 오디오 파일을 로드합니다.

        Args:
            fpath (str): 오디오 파일 경로

        Returns:
            np.ndarray or None: 로드된 오디오 데이터 (numpy 배열) 또는 실패 시 None
        """
        try:
            audio, sr = librosa.load(fpath, sr=self.cfg['sample_rate'])
            return audio
        except Exception as e:
            print(f"오디오 파일 로드 오류 {fpath}: {e}")
            return None

    def _process_file(self, fpath: str, smooth: int=15) -> list:
        """
        단일 오디오 파일을 처리하여 VAD 점수를 생성합니다.
        입력 텐서를 모델/전처리기와 동일한 장치로 이동시킵니다.

        Args:
            fpath (str): 처리할 오디오 파일 경로
            smooth (int): 스무딩 커널 크기

        Returns:
            list: 계산된 프레임별 VAD 결과 
        """
        try:
            wave, sr = librosa.load(fpath, sr=self.cfg['sample_rate'])
            wave_tensor = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).to(self.device)
            wave_len = torch.tensor([wave_tensor.size(-1)], dtype=torch.long).to(self.device)
            # 전처리 (결정된 장치에서 실행됨)
            # self.preprocessor는 초기화 시점에 이미 self.device로 이동됨
            processed_signal, processed_signal_len = self.preprocessor(input_signal=wave_tensor, length=wave_len)
            # 모델 추론 (결정된 장치에서 실행됨, 그래디언트 계산 비활성화)
            # self.model은 초기화 시점에 이미 self.device로 이동되었고, eval/freeze 상태임
            label = self.model(processed_signal, processed_signal_len)
            threshold_low = 0.1
            threshold_high = 0.15
            raw_label = label.squeeze(0).cpu().numpy()
            kernel = np.ones(smooth) / smooth
            if len (raw_label) < smooth:
                label = raw_label
            else:
                label = np.convolve(raw_label, kernel, mode='same')
            candidate_labels = (label > threshold_low).astype(int)
            final_labels = np.zeros_like(candidate_labels)
            # 후보 구간 중 높은 임계값 이상인 확률값을 포함하는 구간만 발화로 인정
            i = 0
            while i < len(candidate_labels):
                if candidate_labels[i] == 1:
                    start_frame_idx = i
                    while i < len(candidate_labels) and candidate_labels[i] == 1:
                        i += 1
                    end_frame_idx = i - 1
                    candidate_segment = label[start_frame_idx:end_frame_idx+1]
                    if np.any(candidate_segment >= threshold_high):
                        final_labels[start_frame_idx:end_frame_idx+1] = 1
                else:
                    i += 1
            return final_labels.tolist()

        except Exception as e:
            print(f"처리 오류 ({os.path.basename(fpath)}): {e.__class__.__name__}: {e}")
            # traceback.print_exc() # 상세 오류 필요시 주석 해제
            return []

    def predict(self, fpaths, smooth=15):
        """
        싱글스레드에서 순차적으로 오디오 파일 목록에 대한 VAD 예측을 수행합니다.

        Args:
            fpaths (str or list[str]): 단일 파일 경로 또는 파일 경로 목록.
            smooth (int): 스무딩 커널의 크기 (홀수 선호).

        Returns:
            list[list[float]]: 각 입력 파일에 대한 VAD 점수 목록. 순서는 입력과 동일. 실패 시 빈 리스트 포함 가능.
        """
        if isinstance(fpaths, str):
            fpaths = [fpaths]
        elif not isinstance(fpaths, list):
            raise TypeError("입력 'fpaths'는 문자열 또는 문자열 리스트여야 합니다.")

        if not fpaths:
            return []

        if smooth % 2 == 0:
            print(f"경고: smooth 값 {smooth}이(가) 짝수입니다. {smooth + 1}(으)로 조정합니다.")
            smooth += 1

        results = []
        print(f"{len(fpaths)}개 파일 처리 시작 (장치: {self.device}, 싱글스레드 모드)...")
        
        # 싱글스레드에서 순차적으로 파일 처리
        for fpath in tqdm(fpaths, desc="VAD 처리 중", unit="파일"):
            result = self._process_file(fpath, smooth)
            results.append(result)
            
        return results


# __main__ 블록: 변경 없음 (초기화 로직이 __init__으로 이동했으므로)
if __name__ == "__main__":
    print("SGVAD 객체 생성 시도...")
    try:
        # SGVAD 초기화 시 내부적으로 장치 결정 및 모델/전처리기 이동이 수행됨
        sgvad = SGVAD()
    except Exception as e:
        print(f"SGVAD 객체 생성 실패: {e}")
        exit(1)
    target_dir = "target_wav"
    if not os.path.isdir(target_dir):
        print(f"오류: 대상 디렉토리 '{target_dir}'를 찾을 수 없습니다.")
        exit(1)
    audio_files = glob.glob(os.path.join(target_dir, "*.wav"))
    if audio_files:
        print(f"\n'{target_dir}'에서 {len(audio_files)}개의 오디오 파일을 찾았습니다.")
        # predict 호출 시 내부 _process_file에서 텐서가 올바른 장치로 이동됨
        all_scores = sgvad.predict(audio_files, smooth=15)
        print("\n--- 예측 결과 요약 ---")
        vad_threshold = 0.5
        processed_count = 0
        failed_count = 0
        for fpath, scores in zip(audio_files, all_scores):
            if scores:
                processed_count += 1
                avg_score = np.mean(scores)
                speech_frames = np.sum(np.array(scores) > vad_threshold)
                total_frames = len(scores)
                speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
                is_speech_present = speech_ratio > 0.1
                label = "음성 가능성 높음" if is_speech_present else "비음성 가능성 높음"
            else:
                failed_count += 1
                print(f"- {os.path.basename(fpath)}: 처리 실패 또는 점수 없음")

        print("-" * 20)
        print(f"총 파일 수: {len(audio_files)}")
        print(f"성공적으로 처리됨: {processed_count}")
        print(f"처리 실패: {failed_count}")
        print("-" * 20)

    else:
        print(f"'{target_dir}' 디렉토리에서 WAV 파일을 찾을 수 없습니다.")
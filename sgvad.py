#sgvad.py
import glob
import torch
from omegaconf import OmegaConf, DictConfig
from nemo.collections.asr.modules import AudioToMFCCPreprocessor, ConvASREncoder
import librosa
import os
import numpy as np

# 멀티프로세싱을 위한 임포트
import multiprocessing
from tqdm import tqdm # 진행률 표시줄
import functools # partial 함수 사용
import traceback # 오류 추적용

# --- GPU 사용 가능 시 멀티프로세싱 시작 방법 설정 ---
# CUDA 사용 시 'spawn'이 'fork'보다 안전할 수 있습니다.
if torch.cuda.is_available():
    try:
        # 이미 설정되지 않은 경우에만 설정 시도
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
            print("CUDA 사용 가능: 멀티프로세싱 시작 방법을 'spawn'으로 설정합니다.")
        elif multiprocessing.get_start_method() != 'spawn':
             print(f"경고: 멀티프로세싱 시작 방법이 이미 '{multiprocessing.get_start_method()}'으로 설정되었습니다. CUDA 사용 시 'spawn'이 권장됩니다.")
    except RuntimeError as e:
        # 이미 시작된 후에는 변경 불가
        print(f"멀티프로세싱 시작 방법 설정 중 오류 (이미 시작된 후 변경 시도?): {e}")
        if multiprocessing.get_start_method() != 'spawn':
             print(f"현재 시작 방법: '{multiprocessing.get_start_method()}'. CUDA 사용 시 'spawn'이 권장됩니다.")


class SGVAD:
    def __init__(self, cfg_path: str="./cfg.yaml"):
        """
        SGVAD 클래스 초기화. 설정 파일과 체크포인트 파일을 로드하여 모델을 준비합니다.
        GPU 사용 가능 여부를 확인하고 설정에 따라 장치를 결정합니다.

        Args:
            cfg_path (str): 설정 파일 (YAML)의 경로. 이 파일 내에 체크포인트 경로와
                            선호하는 device ('cpu' 또는 'cuda:0' 등)가 포함될 수 있습니다.
        """
        print(f"SGVAD 초기화 시작: 설정 파일 '{cfg_path}' 사용")
        try:
            # --- 설정 로드 ---
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {cfg_path}")
            self.cfg = OmegaConf.load(cfg_path) # 설정 로드

            # --- 장치 결정 ---
            requested_device = 'cuda:0' #self.cfg.get('device', None) # 설정 파일에서 device 가져오기 (없으면 None)
            if requested_device and requested_device.startswith('cuda') and not torch.cuda.is_available():
                print(f"경고: 설정 파일에서 CUDA 장치 '{requested_device}'를 요청했지만 CUDA를 사용할 수 없습니다. CPU로 대체합니다.")
                self.device = torch.device('cpu')
            elif requested_device and requested_device.startswith('cuda'):
                try:
                    # 지정된 CUDA 장치 유효성 검사 (예: 'cuda:0')
                    self.device = torch.device(requested_device)
                    # 간단한 텐서 생성으로 실제 사용 가능한지 확인
                    _ = torch.tensor([1.0], device=self.device)
                    print(f"설정 파일에 따라 CUDA 장치 '{self.device}'를 사용합니다.")
                except Exception as e:
                    print(f"경고: 지정된 CUDA 장치 '{requested_device}' 사용 중 오류 발생 ({e}). CPU로 대체합니다.")
                    self.device = torch.device('cpu')
            elif requested_device == 'cpu':
                self.device = torch.device('cpu')
                print("설정 파일에 따라 CPU 장치를 사용합니다.")
            else: # 설정 파일에 device가 없거나 유효하지 않은 경우 자동 감지
                if torch.cuda.is_available():
                    self.device = torch.device('cuda:0') # 기본 CUDA 장치 사용
                    print("CUDA 사용 가능: 기본 CUDA 장치 ('cuda:0')를 사용합니다.")
                else:
                    self.device = torch.device('cpu')
                    print("CUDA 사용 불가: CPU 장치를 사용합니다.")
            # 결정된 장치를 cfg에도 반영 (나중에 참조할 수 있도록)
            self.cfg.device = str(self.device)

            # --- 체크포인트 로드 ---
            if not hasattr(self.cfg, 'ckpt') or not self.cfg.ckpt:
                 raise ValueError(f"{cfg_path} 설정 파일에 'ckpt' 경로가 설정되지 않았습니다.")
            if not os.path.exists(self.cfg.ckpt):
                 raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {self.cfg.ckpt} (설정 파일: {cfg_path})")

            print(f"체크포인트 로드 중: {self.cfg.ckpt}")
            # 중요: 체크포인트를 먼저 CPU로 로드 (GPU 메모리 문제 및 이식성)
            ckpt = torch.load(self.cfg.ckpt, map_location='cpu') # sgvad.pth 에서 로드 

            # 체크포인트 내용 확인
            if 'preprocessor' not in ckpt:
                raise KeyError(f"체크포인트 '{self.cfg.ckpt}'에 'preprocessor' state_dict가 없습니다.")
            if 'vad' not in ckpt:
                raise KeyError(f"체크포인트 '{self.cfg.ckpt}'에 'vad' state_dict가 없습니다.")

            # --- 전처리기 및 모델 초기화 (CPU에서) ---
            print("전처리기 초기화 및 상태 로드 중 (CPU)...")
            # ** 주의: NeMo 모듈 초기화 시 device 인수가 없을 수 있음.
            #    먼저 CPU에서 생성 후 .to(device)로 이동하는 것이 안전함.
            self.preprocessor = AudioToMFCCPreprocessor(**self.cfg.preprocessor)
            self.preprocessor.load_state_dict(ckpt['preprocessor'], strict=True)
            # 디더링 비활성화 및 패딩 설정 적용
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0

            print("VAD 모델 초기화 및 상태 로드 중 (CPU)...")
            self.model = ConvASREncoder(**self.cfg.vad)
            self.model.load_state_dict(ckpt['vad'], strict=True)

            # --- 모델/전처리기 설정 및 결정된 장치로 이동 ---
            self.model.eval() # 평가 모드로 설정
            self.model.freeze() # 모델 가중치 고정 (학습 방지)

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
            audio, sr = librosa.load(fpath, sr=self.cfg.sample_rate)
            return audio
        except Exception as e:
            print(f"오디오 파일 로드 오류 {fpath}: {e}")
            return None

    def _process_file(self, fpath: str, smooth: int) -> list:
        """
        단일 오디오 파일을 처리하여 VAD 점수를 생성합니다. (멀티프로세싱 워커용)
        입력 텐서를 모델/전처리기와 동일한 장치로 이동시킵니다.

        Args:
            fpath (str): 처리할 오디오 파일 경로
            smooth (int): 스무딩 커널 크기

        Returns:
            list: 계산된 프레임별 VAD 점수 리스트. 오류 발생 시 빈 리스트.
        """
        try:
            wave = self.load_audio(fpath)
            if wave is None or len(wave) == 0:
                return []

            # numpy 배열을 float32 타입의 PyTorch 텐서로 변환 (CPU에서)
            wave_tensor_cpu = torch.tensor(wave, dtype=torch.float32).reshape(1, -1)
            wave_len_cpu = torch.tensor([wave_tensor_cpu.size(-1)], dtype=torch.long).reshape(1)

            # --- 입력 텐서를 모델/전처리기가 있는 장치로 이동 ---
            # self.device는 __init__에서 결정된 장치 ('cuda:0' 또는 'cpu')
            wave_tensor = wave_tensor_cpu.to(self.device)
            wave_len = wave_len_cpu.to(self.device)

            # 전처리 (결정된 장치에서 실행됨)
            # self.preprocessor는 초기화 시점에 이미 self.device로 이동됨
            processed_signal, processed_signal_len = self.preprocessor(input_signal=wave_tensor, length=wave_len)

            # 모델 추론 (결정된 장치에서 실행됨, 그래디언트 계산 비활성화)
            # self.model은 초기화 시점에 이미 self.device로 이동되었고, eval/freeze 상태임
            with torch.no_grad():
                mu, _ = self.model(audio_signal=processed_signal, length=processed_signal_len)
                binary_gates = torch.clamp(mu + 0.5, 0.0, 1.0)
                score = binary_gates.sum(dim=1)
                frame_scores = score / 11.0 # 정규화

            # --- 결과를 CPU로 가져와서 NumPy 변환 및 스무딩 ---
            # frame_scores가 GPU에 있을 수 있으므로 .cpu() 호출 후 .numpy()
            frame_scores_np = frame_scores.cpu().numpy().ravel()

            if len(frame_scores_np) > 0:
                kernel = np.ones(smooth) / smooth
                # 컨볼루션 전에 길이 확인
                if len(frame_scores_np) >= smooth:
                     smoothed_scores = np.convolve(frame_scores_np, kernel, mode='same')
                else:
                     # 프레임 수가 커널 크기보다 작을 때 'same' 모드는 작동하나, 결과 해석에 주의
                     # print(f"경고: {os.path.basename(fpath)}의 프레임 수({len(frame_scores_np)})가 스무딩 창({smooth})보다 적습니다.")
                     smoothed_scores = np.convolve(frame_scores_np, kernel, mode='same')
                return smoothed_scores.tolist()
            else:
                return []

        except Exception as e:
            print(f"워커 오류 ({os.path.basename(fpath)}): {e.__class__.__name__}: {e}")
            # traceback.print_exc() # 상세 오류 필요시 주석 해제
            return []

    def predict(self, fpaths, smooth=15, num_workers=None):
        """
        멀티프로세싱을 사용하여 오디오 파일 목록에 대한 VAD 예측을 수행합니다.
        num_workers가 지정되지 않으면 시스템의 CPU 코어 수를 자동으로 사용합니다.

        Args:
            fpaths (str or list[str]): 단일 파일 경로 또는 파일 경로 목록.
            smooth (int): 스무딩 커널의 크기 (홀수 선호).
            num_workers (int, optional): 워커 프로세스 수. 기본값(None)이면 os.cpu_count() // 2 사용.

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

        # 워커 수 결정
        if num_workers is None:
            # GPU 사용 시 메모리 등을 고려하여 CPU 코어의 절반 정도를 기본값으로 사용
            default_workers = max(1, os.cpu_count() // 2)
            num_workers = default_workers
            # print(f"워커 수가 지정되지 않았습니다. 기본값 ({num_workers}) 사용.") # 필요시 주석 해제
        elif num_workers <= 0:
             print(f"경고: 잘못된 num_workers ({num_workers}). 워커 1개 사용.")
             num_workers = 1

        effective_num_workers = min(num_workers, len(fpaths))
        if effective_num_workers == 0: return []

        # 부분 함수 생성 시 self가 자동으로 전달됨
        worker_func = functools.partial(self._process_file, smooth=smooth)

        results = []
        # 멀티프로세싱 풀 컨텍스트 관리자 사용
        try:
            with multiprocessing.Pool(processes=effective_num_workers) as pool:
                print(f"{len(fpaths)}개 파일 처리 시작 (장치: {self.device}, 워커: {effective_num_workers})...")
                results = list(tqdm(pool.imap(worker_func, fpaths), total=len(fpaths), desc="VAD 처리 중", unit="파일"))
        except Exception as e:
            print(f"멀티프로세싱 풀 실행 중 오류 발생: {e}")
            traceback.print_exc()
            # 부분적인 결과라도 반환할 수 있도록 시도 (오류 발생 전까지 처리된 결과)
            # 또는 빈 리스트 반환 결정
            # return [] # 또는 에러 발생 시 빈 결과 반환

        return results


# __main__ 블록: 변경 없음 (초기화 로직이 __init__으로 이동했으므로)
if __name__ == "__main__":
    config_file = "./cfg.yaml"

    print("SGVAD 객체 생성 시도...")
    try:
        # SGVAD 초기화 시 내부적으로 장치 결정 및 모델/전처리기 이동이 수행됨
        sgvad = SGVAD(cfg_path=config_file)
    except (FileNotFoundError, KeyError, ValueError, Exception) as e:
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
        all_scores = sgvad.predict(audio_files, smooth=15) # num_workers=4 등으로 명시적 지정 가능

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
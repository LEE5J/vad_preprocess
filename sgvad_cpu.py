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

if torch.cuda.is_available():
    multiprocessing.set_start_method('spawn', force=True)
class SGVAD:
    # init_from_ckpt 클래스 메서드를 제거하고, 초기화 로직을 __init__으로 통합
    def __init__(self, cfg_path: str="./cfg.yaml"):
        """
        SGVAD 클래스 초기화. 설정 파일과 체크포인트 파일을 로드하여 모델을 준비합니다.

        Args:
            cfg_path (str): 설정 파일 (YAML)의 경로. 이 파일 내에 체크포인트 경로가 포함되어 있어야 합니다.
        """
        print(f"SGVAD 초기화 시작: 설정 파일 '{cfg_path}' 사용")
        try:
            # --- 설정 및 체크포인트 로드 ---
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {cfg_path}")
            self.cfg = OmegaConf.load(cfg_path) # 설정 로드

            # 장치 설정 확인 및 기본값 설정
            if 'device' not in self.cfg:
                self.cfg.device = 'cpu'
                print("경고: cfg.yaml에 'device'가 지정되지 않아 'cpu'로 기본 설정합니다.")
            elif not isinstance(self.cfg.device, str):
                 print(f"경고: cfg.yaml의 device 설정({self.cfg.device})이 올바르지 않을 수 있습니다. 'cpu' 또는 'cuda:0'과 같은 문자열이어야 합니다.")
                 # 필요시 기본값으로 강제 설정 가능: self.cfg.device = 'cpu'

            # 체크포인트 경로 확인
            if not hasattr(self.cfg, 'ckpt') or not self.cfg.ckpt:
                 raise ValueError(f"{cfg_path} 설정 파일에 'ckpt' 경로가 설정되지 않았습니다.")
            if not os.path.exists(self.cfg.ckpt):
                 raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {self.cfg.ckpt} (설정 파일: {cfg_path})")

            print(f"체크포인트 로드 중: {self.cfg.ckpt}")
            # 멀티프로세싱 (특히 fork 사용 시) 문제를 피하기 위해 체크포인트를 CPU로 로드
            ckpt = torch.load(self.cfg.ckpt, map_location='cpu')

            # 체크포인트 내용 확인
            if 'preprocessor' not in ckpt:
                raise KeyError(f"체크포인트 '{self.cfg.ckpt}'에 'preprocessor' state_dict가 없습니다.")
            if 'vad' not in ckpt:
                raise KeyError(f"체크포인트 '{self.cfg.ckpt}'에 'vad' state_dict가 없습니다.")

            # --- 전처리기 및 모델 초기화 ---
            print("전처리기 초기화 및 상태 로드 중...")
            self.preprocessor = AudioToMFCCPreprocessor(**self.cfg.preprocessor)
            self.preprocessor.load_state_dict(ckpt['preprocessor'], strict=True)
            # 디더링 비활성화 및 패딩 설정 적용
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0

            print("VAD 모델 초기화 및 상태 로드 중...")
            self.model = ConvASREncoder(**self.cfg.vad)
            self.model.load_state_dict(ckpt['vad'], strict=True)

            # --- 모델/전처리기 설정 및 장치 이동 ---
            self.model.eval() # 평가 모드로 설정
            self.model.freeze() # 모델 가중치 고정 (학습 방지)

            # 구성 요소들을 설정된 장치로 이동
            # 중요: CPU에서 멀티프로세싱(fork) 사용 시, 이 단계 이후 모델/전처리기 데이터는
            # 자식 프로세스와 Copy-on-Write 방식으로 메모리 공유될 가능성이 높음 (효율적)
            # CUDA 사용 시에는 multiprocessing 방식(fork/spawn) 및 장치 관리에 주의 필요
            print(f"모델 및 전처리기를 장치 '{self.cfg.device}'로 이동 중...")
            self.preprocessor.to(self.cfg.device)
            self.model.to(self.cfg.device)

            print("SGVAD 초기화 완료.")

        except FileNotFoundError as e:
            print(f"초기화 오류: 필수 파일 누락 - {e}")
            raise # 오류를 다시 발생시켜 호출자에게 알림
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
            # librosa를 사용하여 오디오 로드 (설정된 샘플링 속도로)
            audio, sr = librosa.load(fpath, sr=self.cfg.sample_rate)
            return audio
        except Exception as e:
            print(f"오디오 파일 로드 오류 {fpath}: {e}")
            # 실패를 알리기 위해 None 반환, _process_file에서 처리됨
            return None

    def _process_file(self, fpath: str, smooth: int) -> list:
        """
        단일 오디오 파일을 처리하여 VAD 점수를 생성합니다. (멀티프로세싱 워커용)

        Args:
            fpath (str): 처리할 오디오 파일 경로
            smooth (int): 스무딩 커널 크기

        Returns:
            list: 계산된 프레임별 VAD 점수 리스트. 오류 발생 시 빈 리스트.
        """
        try:
            wave = self.load_audio(fpath)
            if wave is None or len(wave) == 0: # 로딩 오류 또는 빈 오디오 처리
                # print(f"비어 있거나 읽을 수 없는 파일 건너뛰기: {fpath}") # 너무 많은 로그를 피하기 위해 주석 처리 가능
                return [] # 이 파일에 대해 빈 리스트 반환

            # numpy 배열을 float32 타입의 PyTorch 텐서로 변환
            wave_tensor = torch.tensor(wave, dtype=torch.float32).reshape(1, -1)
            wave_len = torch.tensor([wave_tensor.size(-1)], dtype=torch.long).reshape(1)

            # 워커 프로세스 내에서도 텐서를 모델/전처리기와 동일한 장치로 이동
            wave_tensor = wave_tensor.to(self.cfg.device)
            wave_len = wave_len.to(self.cfg.device)

            # 전처리
            # self.preprocessor는 초기화 시점에 장치로 이동되었음
            processed_signal, processed_signal_len = self.preprocessor(input_signal=wave_tensor, length=wave_len)

            # 모델 추론 (그래디언트 계산 비활성화)
            # self.model은 초기화 시점에 장치로 이동되었고, eval/freeze 상태임
            with torch.no_grad():
                mu, _ = self.model(audio_signal=processed_signal, length=processed_signal_len)
                binary_gates = torch.clamp(mu + 0.5, 0.0, 1.0)
                score = binary_gates.sum(dim=1)
                frame_scores = score / 11.0 # 정규화 (cfg에서 관리하는 것이 좋을 수 있음)

            # numpy로 변환 및 스무딩
            if isinstance(frame_scores, torch.Tensor):
                frame_scores_np = frame_scores.cpu().numpy().ravel()
            else:
                frame_scores_np = np.array(frame_scores).ravel()

            if len(frame_scores_np) > 0:
                kernel = np.ones(smooth) / smooth
                if len(frame_scores_np) >= smooth:
                     smoothed_scores = np.convolve(frame_scores_np, kernel, mode='same')
                else:
                     # print(f"경고: {os.path.basename(fpath)}의 프레임 수({len(frame_scores_np)})가 스무딩 창({smooth})보다 적습니다. 'same' 컨볼루션 사용.")
                     smoothed_scores = np.convolve(frame_scores_np, kernel, mode='same')
                return smoothed_scores.tolist()
            else:
                return []

        except Exception as e:
            # 워커 내 오류 로깅 개선
            print(f"워커 오류 ({os.path.basename(fpath)}): {e.__class__.__name__}: {e}")
            # traceback.print_exc() # 상세 오류 필요시 주석 해제
            return [] # 실패 시 빈 리스트 반환


    def predict(self, fpaths, smooth=15, num_workers=None):
        """
        멀티프로세싱을 사용하여 오디오 파일 목록에 대한 VAD 예측을 수행합니다.
        num_workers가 지정되지 않으면 시스템의 CPU 코어 수를 자동으로 사용합니다.

        Args:
            fpaths (str or list[str]): 단일 파일 경로 또는 파일 경로 목록.
            smooth (int): 스무딩 커널의 크기 (홀수 선호).
            num_workers (int, optional): 워커 프로세스 수. 기본값(None)이면 os.cpu_count() 사용.

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
            num_workers = os.cpu_count()//2
            # print(f"워커 수가 지정되지 않았습니다. 시스템 CPU 코어 수({num_workers})를 사용합니다.") # 필요시 주석 해제
        elif num_workers <= 0:
             print(f"경고: 잘못된 num_workers ({num_workers}). 워커 1개 사용.")
             num_workers = 1
        # else:
            # print(f"지정된 워커 수 사용: {num_workers}") # 필요시 주석 해제

        # 실제 사용할 워커 수는 처리할 파일 수와 워커 수 중 작은 값
        effective_num_workers = min(num_workers, len(fpaths))
        if effective_num_workers == 0: return [] # 처리할 파일이 없으면 즉시 반환

        # 부분 함수 생성 시 self가 자동으로 전달됨 (self._process_file은 인스턴스 메서드)
        worker_func = functools.partial(self._process_file, smooth=smooth)

        results = []
        # 멀티프로세싱 풀 컨텍스트 관리자 사용
        with multiprocessing.Pool(processes=effective_num_workers) as pool:
            print(f"{len(fpaths)}개 파일 처리 시작 (워커: {effective_num_workers})...")
            # tqdm으로 진행률 표시
            results = list(tqdm(pool.imap(worker_func, fpaths), total=len(fpaths), desc="VAD 처리 중", unit="파일"))
            # print("처리 완료.") # tqdm이 완료 메시지를 포함하므로 중복될 수 있음

        return results


# __main__ 블록: 변경된 초기화 방식 사용
if __name__ == "__main__":
    # 설정 파일 경로
    config_file = "./cfg.yaml"

    print("SGVAD 객체 생성 시도...")
    try:
        # 이제 __init__을 직접 호출하여 인스턴스 생성
        sgvad = SGVAD(cfg_path=config_file)
    except (FileNotFoundError, KeyError, ValueError, Exception) as e:
        # __init__ 내부에서 발생한 오류 처리
        print(f"SGVAD 객체 생성 실패: {e}")
        # 상세 오류는 __init__ 내부에서 이미 출력되었을 수 있음
        exit(1) # 오류 발생 시 종료

    # --- 오디오 파일 찾기 ---
    target_dir = "target_wav" # 대상 디렉토리
    if not os.path.isdir(target_dir):
        print(f"오류: 대상 디렉토리 '{target_dir}'를 찾을 수 없습니다.")
        exit(1)

    audio_files = glob.glob(os.path.join(target_dir, "*.wav")) # WAV 파일 목록 가져오기

    # --- 예측 수행 ---
    if audio_files:
        print(f"\n'{target_dir}'에서 {len(audio_files)}개의 오디오 파일을 찾았습니다.")
        all_scores = sgvad.predict(audio_files, smooth=15)

        print("\n--- 예측 결과 요약 ---")
        vad_threshold = 0.5 # 예시 임계값
        processed_count = 0
        failed_count = 0

        # 결과 분석 및 출력
        for fpath, scores in zip(audio_files, all_scores):
            if scores: # 점수가 비어있지 않으면 성공으로 간주
                processed_count += 1
                avg_score = np.mean(scores)
                speech_frames = np.sum(np.array(scores) > vad_threshold)
                total_frames = len(scores)
                speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
                is_speech_present = speech_ratio > 0.1
                label = "음성 가능성 높음" if is_speech_present else "비음성 가능성 높음"
                # 상세 결과 출력 (필요시 주석 해제)
                # print(f"- {os.path.basename(fpath)}: 프레임={total_frames}, 평균 점수={avg_score:.3f}, 음성비율={speech_ratio:.2f} -> {label}")
            else: # 점수가 비어있으면 실패로 간주
                failed_count += 1
                print(f"- {os.path.basename(fpath)}: 처리 실패 또는 점수 없음")
        print("-" * 20)

    else: # 오디오 파일을 찾지 못한 경우
        print(f"'{target_dir}' 디렉토리에서 WAV 파일을 찾을 수 없습니다.")
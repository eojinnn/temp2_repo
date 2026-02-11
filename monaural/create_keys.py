import os
import math
import wave
import contextlib
import shutil
from tqdm import tqdm

def create_keys(audio_dir, output_keys_file, segment_len_sec=10.0, fs=24000):
    """
    오디오 폴더를 스캔하여 LMDB 없이 사용할 수 있는 keys.txt를 생성합니다.
    
    :param audio_dir: .wav 파일이 있는 최상위 폴더 (dataset root/foa_dev 등)
    :param output_keys_file: 저장할 keys.txt 경로
    :param segment_len_sec: 자를 구간의 길이 (초). convert_lmdb.py에서는 기본 10초를 사용함.
                            Dataset_loader에서 설정한 값과 일치해야 함.
    :param fs: 샘플링 레이트 (프레임 수 계산용)
    """
    output_dir = os.path.dirname(output_keys_file)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Generatings keys to {output_keys_file}...")

    with open(output_keys_file, 'w') as f_keys:
        
        # 오디오 디렉토리 내의 모든 파일 탐색 (서브 디렉토리 포함)
        audio_files = []
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        # 파일 순회
        for wav_path in tqdm(audio_files):
            file_name = os.path.basename(wav_path)
            base_name = os.path.splitext(file_name)[0] # 확장자 제거 (예: fold1_room1_mix001)
            
            # 오디오 파일 길이(샘플 수) 확인
            try:
                with contextlib.closing(wave.open(wav_path, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    
                    # 만약 fs가 다르다면 비율 조정 필요 (여기선 fs가 맞다고 가정)
                    if rate != fs:
                        # 경고: 실제 데이터 fs와 설정된 fs가 다르면 길이 계산 오차 발생 가능
                        pass
            except Exception as e:
                print(f"Error reading {wav_path}: {e}")
                continue

            # 세그먼트 개수 계산 (convert_lmdb.py 로직 모방)
            # convert_lmdb.py: segment_num = math.ceil(data_frame_num / segment_data_frame_num)
            
            samples_per_segment = int(segment_len_sec * fs)
            total_segments = math.ceil(frames / samples_per_segment)
            
            # 각 세그먼트 별 키 생성 및 기록
            for seg_id in range(total_segments):
                # 포맷: 파일명_seg_총개수_현재인덱스
                # 예: fold1_room1_mix001_seg_6_0
                key = f"{base_name}_seg_{total_segments}_{seg_id}"
                f_keys.write(f"{key}\n")

    print(f"Done! Keys saved to {output_keys_file}")


if __name__ == "__main__":
    target_audio_dir = 'F:/interspeech2026/2024DCASE_data/mic_dev' 
    save_keys_path = './data/keys.txt'
    seg_len = 10.0 
    
    create_keys(
        audio_dir=target_audio_dir, 
        output_keys_file=save_keys_path, 
        segment_len_sec=seg_len, 
        fs=24000
    )
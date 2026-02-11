import os
import numpy as np
path = "F:/interspeech2026/Solution_on_3D_SELD/data/feature_labels_2023/mic_dev_label"
label_files = [f for f in os.listdir(path) if f.endswith('.npy')]
print(f"폴더 안에 총 {len(label_files)}개의 npy 파일이 있습니다.")

# 첫 번째 파일만 샘플로 열어서 확인해보기
if len(label_files) > 0:
    sample_filename = label_files[0]
    full_path = os.path.join(path, sample_filename)
    
    try:
        # numpy 파일 로드
        data = np.load(full_path)
        
        print(f"\n[{sample_filename}] 파일 정보:")
        print(f"1. Shape (형상): {data.shape}")  # 예: (500, 14) 또는 (50, 4) 등
        print(f"2. Dtype (데이터 타입): {data.dtype}")
        print("-" * 30)
        print("3. 데이터 내용 (상위 5개 행):")
        print(data[:5]) # 앞부분 5줄만 출력
        print("-" * 30)
        
        # 만약 전체가 0인지, 값이 들어있는지 확인하고 싶다면:
        print(f"Min 값: {data.min()}, Max 값: {data.max()}")
        
    except Exception as e:
        print(f"파일을 읽는 중 에러 발생: {e}")
else:
    print("해당 경로에 .npy 파일이 없습니다.")

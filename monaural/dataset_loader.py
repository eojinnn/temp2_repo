import os
import torch
import time
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset

class Dataset_loader(Dataset):
    def __init__(self, keys_file, audio_dir, label_dir, segment_len_sec=5, fs=24000, split=None, data_process_fn=None):
        super().__init__()
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.split = split
        self.fs = fs
        
        #5sec
        self.segment_len_sec = segment_len_sec
        self.segment_len = int(segment_len_sec * fs) # 120,000 samples
        
        #sincnet
        self.win_len = 480        # 24000*0.02s = 480 samples
        self.hop_len = self.win_len # no overlap
        self.target_seq_len = 250   # 5초/0.02s = 250 frames

        self.data_process_fn = data_process_fn        
        self.label_fps = 10
        self.data_items= []

        valid_files_count = 0
        for sub_folder in os.listdir(self.audio_dir):
            loc_aud_folder = os.path.join(self.audio_dir, sub_folder)
            if os.path.isdir(loc_aud_folder):
                for file_name in os.listdir(loc_aud_folder):
                    if file_name.endswith('.wav'):
                        try:
                            # 'fold1' -> '1' 추출
                            fold_str = file_name.split('_')[0] 
                            fold_num = int(fold_str.replace('fold', ''))
                        except ValueError:
                            print(f"Skipping {file_name}: Cannot parse fold number.")
                            continue

                        if self.split is not None and fold_num not in self.split:
                            continue

                        base_name = file_name.split('.')[0] 
                        wav_path = os.path.join(loc_aud_folder, file_name)
                        # self.file_path_map[base_name] = wav_path

                        try:
                            with sf.SoundFile(wav_path) as f:
                                frames = f.frames
                                samplerate = f.samplerate
                            
                            duration_sec = frames / samplerate
                            
                            # 5초 단위로 몇 조각이 나오는지 계산 (올림 처리)
                            # 예: 60초 -> 12개, 10초 -> 2개
                            num_segments = int(np.ceil(duration_sec / self.segment_len_sec))
                            
                            for i in range(num_segments):
                                start_time = i * self.segment_len_sec
                                # 리스트에 정보 저장: (전체경로, 시작시간, 파일이름)
                                self.data_items.append((wav_path, start_time, base_name))
                            
                        except Exception as e:
                            print(f"Error reading {file_name}: {e}")

        print(f"Found {valid_files_count} valid files based on split {self.split}.")
        print(f"Generated {len(self.data_items)} training segments (5 sec each).")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, index):
        wav_path, start_time, base_name = self.data_items[index]
        
        with sf.SoundFile(wav_path) as f:
            orig_sr = f.samplerate
            frame_offset = int(start_time * orig_sr)
            num_frames = int(self.segment_len_sec * orig_sr)
            
            # Seek & Read (범위 체크)
            if frame_offset >= f.frames:
                data_np = np.zeros((f.channels, num_frames), dtype=np.float32)
            else:
                f.seek(frame_offset)
                frames_to_read = min(num_frames, f.frames - frame_offset)
                data_read = f.read(frames=frames_to_read, dtype='float32', always_2d=True)
                data_np = data_read.T # (Time, Ch) -> (Ch, Time)

        data = torch.from_numpy(data_np).float()

        # 부족한 길이 패딩 (파일 끝부분 처리)
        if data.shape[1] < num_frames: 
            pad_amount = num_frames - data.shape[1]
            data = F.pad(data, (0, pad_amount), mode='constant', value=0)

        # 리샘플링 (필요시) - torchaudio 의존성 제거됨
        if orig_sr != self.fs:
             data = F.interpolate(data.unsqueeze(0), size=self.segment_len, mode='linear', align_corners=False).squeeze(0)

        # 7. 최종 길이 맞추기
        if data.shape[1] < self.segment_len:
            pad_amount = self.segment_len - data.shape[1]
            data = F.pad(data, (0, pad_amount), mode='constant', value=0)
        elif data.shape[1] > self.segment_len:
            data = data[:, :self.segment_len]

        # -----------------------------------------------------------
        # Windowing (Framing)
        # Input Shape: (Channel, 240000) -> Output Shape: (Channel, 500, 1023)
        # -----------------------------------------------------------
        # (Channel, Time, Window) -> (4, 250, 480)
        data_framed = data.unfold(-1, self.win_len, self.hop_len)
        
        if data_framed.shape[1] > self.target_seq_len:
            data_framed = data_framed[:, :self.target_seq_len, :]
            
        # -----------------------------------------------------------
        #label
        label_path = os.path.join(self.label_dir, f"{base_name}.npy")
        target_label_len = int(self.segment_len_sec * self.label_fps) # 예: 5.0 * 10 = 50 frames

        if os.path.exists(label_path):
            full_label = np.load(label_path, mmap_mode='r') # Shape: (Total_Frames, Classes)
            # 2. 시작 프레임 계산
            # start_time(초) * FPS = 시작 프레임 인덱스
            start_frame = int(start_time * self.label_fps)
            end_frame = start_frame + target_label_len
            
            # 3. 라벨 슬라이싱 (범위 체크)
            if start_frame >= full_label.shape[0]:
                label = np.zeros((target_label_len, full_label.shape[1]), dtype=np.float32)
            elif end_frame > full_label.shape[0]:
                valid = full_label[start_frame:, :]
                # mmap 객체에서 복사본 생성
                valid = np.array(valid) 
                pad = target_label_len - valid.shape[0]
                label = np.pad(valid, ((0, pad), (0, 0)), mode='constant')
            else:
                label = np.array(full_label[start_frame:end_frame, :])
        else:
            print('zero labels error')
            label = np.zeros((target_label_len, 13), dtype=np.float32)
   
        label_tensor = torch.from_numpy(label).float()

        seg_id = int(start_time // self.segment_len_sec)
        unique_name = f"{base_name}_seg_{seg_id}"

        # Data Process (Mixup 등)
        if self.data_process_fn is not None:
            # 필요에 따라 data_process_fn의 인자를 맞춰주세요.
            # 예: data, label_tensor = self.data_process_fn(data, label_tensor)
            pass
        
        return {'input': data_framed, 'target': label_tensor, 'wav_name': unique_name}

    def collater(self, samples):
        # Stack inputs: (Batch, Channel, Time, Window_Len)
        inputs = torch.stack([s['input'] for s in samples], dim=0)
        targets = torch.stack([s['target'] for s in samples], dim=0)
        wav_names = [s['wav_name'] for s in samples]
        
        return {
            'input': inputs,
            'target': targets,
            'wav_names': wav_names
        }
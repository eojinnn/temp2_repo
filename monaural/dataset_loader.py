import os
import torch
import time
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset

class Dataset_loader(Dataset):
    def __init__(self, keys_file, audio_dir, label_dir, segment_len_sec=10, fs=24000, split=None, data_process_fn=None):
        super().__init__()
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.split = split
        self.fs = fs
        self.segment_len_sec = segment_len_sec
        self.segment_len = int(segment_len_sec * fs) # 240,000 samples
        self.data_process_fn = data_process_fn        
        self.env = None
        self.label_fps = 10
        
        #sincnet
        self.win_len = 1023         #ngcc
        self.target_seq_len = 500   #목표 시퀀스 길이: 10초 기준 약 500프레임 (20ms=0.02s)
        self.hop_len = self.segment_len // self.target_seq_len #480 samples

        self.file_path_map = {}
        for sub_folder in os.listdir(self.audio_dir):
            loc_aud_folder = os.path.join(self.audio_dir, sub_folder)
            if os.path.isdir(loc_aud_folder):
                for file_name in os.listdir(loc_aud_folder):
                    if file_name.endswith('.wav'):
                        base_name = file_name.split('.')[0] 
                        wav_path = os.path.join(loc_aud_folder, file_name)
                        self.file_path_map[base_name] = wav_path
        # print(f"Found {len(self.file_path_map)} wav files.")

        self.keys = []
        with open(keys_file, 'r') as f:
            lines = f.readlines()
            for k in lines:
                k = k.strip()
                if self.split is not None:
                    if int(k.split('_')[0][4:]) in self.split:
                        self.keys.append(k)
                else:
                    self.keys.append(k)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        t0 = time.time()
        key = self.keys[index]
        parts = key.split('_seg_') 
        
        wav_name = parts[0]   # 예: "fold3_room12_mix002"
        seg_info = parts[1].split('_') 
        seg_id = int(seg_info[1]) 
        
        wav_path = self.file_path_map.get(wav_name)
        start_time = seg_id * self.segment_len_sec 
        
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
                # always_2d=True로 채널 차원 확보
                data_read = f.read(frames=frames_to_read, dtype='float32', always_2d=True)
                data_np = data_read.T # (Time, Ch) -> (Ch, Time)

        # 4. 텐서 변환
        data = torch.from_numpy(data_np).float()

        # 5. 부족한 길이 패딩 (파일 끝부분 처리)
        if data.shape[1] < num_frames:
            pad_amount = num_frames - data.shape[1]
            data = F.pad(data, (0, pad_amount), mode='constant', value=0)

        # 6. 리샘플링 (필요시) - torchaudio 의존성 제거됨
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
        
        # Unfold를 위해 필요한 패딩 계산
        # 공식: Output_Size = (Input + 2*pad - dilation*(kernel-1) - 1)/stride + 1
        # 우리가 원하는 Output_Size는 self.target_seq_len (500)
        
        required_len = (self.target_seq_len - 1) * self.hop_len + self.win_len
        if data.shape[1] < required_len:
            data = F.pad(data, (0, required_len - data.shape[1]), mode='reflect')
        
        # unfold(dimension, size, step)
        # 결과: (Channel, Window_Len, Time_Frames)
        data_framed = data.unfold(-1, self.win_len, self.hop_len)
        
        # 만약 프레임이 조금 더 나왔으면 자름
        if data_framed.shape[1] > self.target_seq_len:
            data_framed = data_framed[:, :self.target_seq_len, :]
            
        # -----------------------------------------------------------
        #label
        label_path = os.path.join(self.label_dir, f"{wav_name}.npy")
        target_label_len = int(self.segment_len_sec * self.label_fps) # 예: 2.0 * 10 = 20 frames

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

        # Padding (혹시 모를 차원 오류 방지 - 기존 로직 유지)
        if label.shape[0] < target_label_len:
            pad_h = target_label_len - label.shape[0]
            label = np.pad(label, ((0, pad_h), (0, 0)), mode='constant')
        elif label.shape[0] > target_label_len:
            label = label[:target_label_len, :]
   
        label_tensor = torch.from_numpy(label).float()

        # Data Process (Mixup 등)
        if self.data_process_fn is not None:
            # 필요에 따라 data_process_fn의 인자를 맞춰주세요.
            # 예: data, label_tensor = self.data_process_fn(data, label_tensor)
            pass
        
        return {'input': data_framed, 'target': label_tensor, 'wav_name': key}

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
import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class SalsaliteEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        # 1. 파라미터 로드
        self.fs = params['fs']
        self.hop_len_s = params['hop_len_s']
        # SALSA-Lite 관련 파라미터가 params에 있어야 합니다.
        # 없다면 기본값(DCASE2023 등)을 사용하거나 params에 추가해야 합니다.
        self.fmin_doa = params.get('fmin_doa_salsalite', 50)
        self.fmax_doa = params.get('fmax_doa_salsalite', 2000)
        self.fmax_spectra = params.get('fmax_spectra_salsalite', 9000)
        self.nb_mel_bins = params.get('nb_mel_bins', 64) # Salsalite 사용 시 Bin 수가 달라질 수 있음

        # 2. STFT 파라미터 설정
        self.hop_len = int(self.fs * self.hop_len_s)
        self.win_len = 2 * self.hop_len
        self.nfft = self._next_greater_power_of_2(self.win_len)
        self.window = torch.hann_window(self.win_len)
        
        # 3. Frequency Bin 인덱스 계산 (cls_feature_class.py 로직)
        self.lower_bin = int(np.floor(self.fmin_doa * self.nfft / float(self.fs)))
        self.lower_bin = max(1, self.lower_bin)
        
        self.upper_bin = int(np.floor(min(self.fmax_doa, self.fs//2) * self.nfft / float(self.fs)))
        self.cutoff_bin = int(np.floor(self.fmax_spectra * self.nfft / float(self.fs)))
        
        # Upper bin이 cutoff보다 큰지 확인 (assert)
        if self.upper_bin > self.cutoff_bin:
             print(f"Warning: upper_bin({self.upper_bin}) > cutoff_bin({self.cutoff_bin}). Clipping upper_bin.")
             self.upper_bin = self.cutoff_bin

        # 실제 출력될 Feature의 차원 (Freq 축 크기)
        # cls_feature_class.py: self._nb_mel_bins = self._cutoff_bin - self._lower_bin 
        self.out_bins = self.cutoff_bin - self.lower_bin

        # 4. 정규화 상수 계산 (Delta & Freq Vector)
        c = 343.0
        self.delta = 2 * np.pi * self.fs / (self.nfft * c)
        
        # Freq Vector: [0, 1, 2, ..., nfft//2]
        freq_vector = torch.arange(self.nfft // 2 + 1, dtype=torch.float32)
        freq_vector[0] = 1.0 # 0번 빈은 1로 설정 (Div by Zero 방지)
        
        # Broadcasting을 위해 Shape 맞춤: (1, 1, Freq, 1) -> (B, C, F, T)와 연산
        self.register_buffer('freq_vector', freq_vector.view(1, 1, -1, 1))
        self.register_buffer('delta_tensor', torch.tensor(self.delta))

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def forward(self, x):
        # x: Raw Audio (Batch, 4, Samples)
        B, C, L = x.shape
        device = x.device
        if self.window.device != device:
            self.window = self.window.to(device)
            
        # 1. STFT 수행 (Batch*Channel 병렬 처리)
        x_reshaped = x.view(B*C, L)
        stft = torch.stft(x_reshaped, n_fft=self.nfft, hop_length=self.hop_len, 
                          win_length=self.win_len, window=self.window, return_complex=True)
        
        # (Batch, Channel, Freq, Time) 형태로 복구
        _, F_dim, T_dim = stft.shape
        linear_spectra = stft.view(B, C, F_dim, T_dim)
        
        # -----------------------------------------------------------
        # [핵심] _get_salsalite 로직 구현
        # -----------------------------------------------------------
        
        # Reference Channel (0번)과 나머지 채널 (1,2,3번) 분리
        ref_channel = linear_spectra[:, 0:1, :, :]      # (B, 1, F, T)
        other_channels = linear_spectra[:, 1:, :, :]    # (B, 3, F, T)
        
        # 1. Cross-Spectrum & Phase Calculation
        # phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        # PyTorch: (B, 3, F, T) * (B, 1, F, T) (Broadcasting)
        cross_spectra = other_channels * torch.conj(ref_channel)
        phase_vector = torch.angle(cross_spectra)
        
        # 2. Frequency Normalization
        # phase_vector = phase_vector / (self._delta * self._freq_vector)
        phase_vector = phase_vector / (self.delta_tensor * self.freq_vector)
        
        # 3. Frequency Cropping (Bandwidth 제한)
        # phase_vector = phase_vector[:, self._lower_bin:self._cutoff_bin, :]
        # PyTorch Freq axis is dim 2
        phase_vector = phase_vector[:, :, self.lower_bin:self.cutoff_bin, :]
        
        # 4. Upper Bin Masking (고주파 노이즈 제거)
        # phase_vector[:, self._upper_bin:, :] = 0
        # 주의: 여기 self.upper_bin은 '원본 인덱스' 기준이므로, 잘린 벡터(Cropped)에서는 오프셋을 빼줘야 함
        # 잘린 벡터의 인덱스 0 = 원본의 lower_bin
        mask_start_idx = self.upper_bin - self.lower_bin
        if mask_start_idx < phase_vector.shape[2]:
            phase_vector[:, :, mask_start_idx:, :] = 0.0
            
        # 5. Shape Adjustment for Model Input
        # (Batch, 3, Freq_cropped, Time) -> (Batch, 3, Time, Freq_cropped)
        # ResNet Backbone은 (B, C, T, F) 형태를 기대하므로 Permute 수행
        # Salsalite Feature는 3채널(1-0, 2-0, 3-0)입니다.
        phase_vector = phase_vector.permute(0, 1, 3, 2)
        
        return phase_vector

        
# ==============================================================================
# 2. GCC-PHAT Module (PyTorch Implementation of provided NumPy code)
# ==============================================================================
class GCCPHATEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.fs = params['fs']
        self.nb_mel_bins = params['nb_mel_bins'] # 보통 64
        
        # 데이터 로더에서 넘어오는 윈도우 크기 (1023)
        # FFT를 위해 2의 제곱수인 1024로 설정
        self.win_len = 1023 
        self.nfft = 1024 
        
        # 4채널 마이크의 가능한 모든 쌍 (6개)
        # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        self.pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        
        # Hann Window (미리 생성해서 버퍼에 등록 - GPU 자동 이동)
        self.register_buffer('window', torch.hann_window(self.win_len))

    def forward(self, x):
        """
        Input x: (Batch, Channel, Time, Window)
                 예: (4, 4, 500, 1023)
        """
        B, C, T, W = x.shape
        
        # 1. Window Function 적용
        # (B, C, T, W) * (W,)
        x = x * self.window
        
        # 2. RFFT 수행 (Real-to-Complex FFT)
        # 마지막 차원(W=1023)에 대해 수행 -> 결과는 1024//2 + 1 = 513개 주파수 빈
        # n=1024로 지정하면 자동으로 Zero-padding 됨
        X = torch.fft.rfft(x, n=self.nfft, dim=-1) # (B, C, T, 513)
        
        gcc_features = []
        
        # 3. 각 마이크 쌍에 대해 GCC-PHAT 계산
        for (m1, m2) in self.pairs:
            # Cross-Power Spectrum: X1 * conj(X2)
            X1 = X[:, m1, :, :]
            X2 = X[:, m2, :, :]
            R = X1 * torch.conj(X2)
            
            # PHAT Weighting: 진폭으로 나누어 정규화 (위상 정보만 남김)
            # 1e-6은 0으로 나누기 방지
            R_phat = R / (torch.abs(R) + 1e-6)
            
            # IFFT로 시간 지연(Lag) 도메인으로 변환
            # 결과 Shape: (B, T, nfft=1024)
            cc = torch.fft.irfft(R_phat, n=self.nfft, dim=-1)
            
            # 4. Shift & Crop (중요: 0지연을 중앙으로)
            # irfft 결과는 [0, 1, ... , -1] 순서이므로 이를 중앙 정렬해야 함.
            # 우리는 nb_mel_bins(64)만큼만 필요하므로, 양쪽 끝에서 32개씩 가져와서 붙임.
            
            shift_len = self.nb_mel_bins // 2  # 32
            
            # 뒤쪽 32개 (Negative Lags) + 앞쪽 32개 (Positive Lags)
            cc_crop = torch.cat([cc[..., -shift_len:], cc[..., :shift_len]], dim=-1)
            
            gcc_features.append(cc_crop)
            
        # 5. 최종 스택
        # list of (B, T, 64) -> (B, 6, T, 64)
        gcc_out = torch.stack(gcc_features, dim=1) 
        
        # 6. ResNet 입력을 위한 차원 변경
        # ResNet은 (Batch, Channel, Freq, Time) 순서를 좋아함
        # (B, 6, T, 64) -> (B, 6, 64, 500)
        gcc_out = gcc_out.permute(0, 1, 3, 2)
        
        return gcc_out
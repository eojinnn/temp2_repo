import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

# 기존 ResNet, Conformer 모듈 import
from models.resnet import resnet18, resnet18_nopool, BasicBlock
from .doa_feat import SalsaliteEncoder, GCCPHATEncoder
from models.conformer import ConformerBlock
from .dnn_models import SincNet

class ResnetConformer_2026(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, fs=24000, sig_len=480, params=None):
        super().__init__()
        self.fs = fs
        self.sig_len = sig_len
        # --- A. SincNet Encoder Setup ---
        # Note: cnn_N_filt[0] is set to 64 to match GCC bins (was 128 in param dict)
        # If you want to keep 128, you must change GCC bins or handle concat mismatch.
        # Here we assume concatenation along channel, so Freq dim (64) must match.
        sincnet_params = {
            'input_dim': sig_len,
            'fs': fs,
            'cnn_N_filt': [in_dim,in_dim,in_dim,in_dim], #32
            'cnn_len_filt': [sig_len-1, 11, 9, 7],
            'cnn_max_pool_len': [2, 2, 2, 2],
            'cnn_use_laynorm_inp': False,
            'cnn_use_batchnorm_inp': False,
            'cnn_use_laynorm': [False, False, False, False],
            'cnn_use_batchnorm': [True, True, True, True],
            'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'linear'],
            'cnn_drop': [0.0, 0.0, 0.0, 0.0],
            'use_sinc': True,
        }
        self.encoder = SincNet(sincnet_params)
        
        self.aux_feat = GCCPHATEncoder(params=params, win_len=sig_len)
        # self.aux_feat = SalsaliteEncoder(params=params)
        
        # --- C. Backbone Setup ---
        # SincNet(4ch) + GCC(6ch) = 10 Channels input
        self.resnet = resnet18_nopool(in_channel=in_channel)
        
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = 8
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13),
            # nn.Sigmoid()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim),
            nn.Tanh()
        ) 

    def forward(self, x):
        def sync_time():                                                                                #time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.time()
        """
        x: Framed Audio Input from Dataset_loader
           Shape: (Batch, Channel, Time, Window_Len) -> (B, 4, 250, 480)
        """
        B, C, T, W = x.shape

        t_before= time.time()
        t0 = sync_time()       
                                                                                 #time
        x_sinc_in = x.reshape(B * C, 1, T * W)
        sinc_feat = self.encoder(x_sinc_in)  

        t1 = sync_time()                  
                                                                      #time                
        # (B*C, 128, 7500) -> (B*C, 128, 250) 로 줄임 (Adaptive Pooling)
        sinc_feat_pooled = F.adaptive_avg_pool1d(sinc_feat, T)
        # (B*C*T, F) -> (B, C, T, F)
        _, C_sinc, T_sinc = sinc_feat_pooled.shape
        sinc_feat = sinc_feat_pooled.reshape(B, C, C_sinc, T_sinc)
        # ResNet에 넣기 위해 (B, Channel, Time, Freq) 형태로 변환해야 함
        # 현재 T_sinc(7500)는 너무 길고, ResNet은 2D 이미지를 원함.
        # NGCC는 이 시점에서 GCC를 구하거나 Pooling을 더 함.
        # 여기서는 ResNet 입력을 위해 [B, C_sinc, T_target, F_fake] 처럼 만들거나
        # Time 축을 원래 T(250)에 맞게 보간(Interpolate)합니다.

        # GCC: (B, 6, Time, Freq)
        gcc_feat = self.aux_feat(x)

        t2 = sync_time()

        features = torch.cat([sinc_feat, gcc_feat], dim=1)
        features = features.permute(0, 1, 3, 2)  # (B, 10, 64,250)
        # 2. Backbone Forward
        conv_outputs = self.resnet(features)

        t3 = sync_time()

        N, C_out, T_out, W_out = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T_out, C_out * W_out)
        conformer_outputs = self.input_projection(conv_outputs)
        
        for layer in self.conformer_layers:
            conformer_outputs = layer(conformer_outputs)

        t4 = sync_time()

        outputs = conformer_outputs.permute(0, 2, 1)
        outputs = self.t_pooling(outputs)
        outputs = outputs.permute(0, 2, 1)
        
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)

        pred = torch.cat((sed, doa), dim=-1)
        
        target_frames = 100
        if pred.shape[1] != target_frames:
            # interpolate는 (Batch, Channel, Time) 순서를 원하므로 뒤집어야 함
            # (Batch, Time, Class) -> (Batch, Class, Time)
            pred = pred.transpose(1, 2)
            pred = F.interpolate(pred, size=target_frames, mode='linear', align_corners=False)   
            pred = pred.transpose(1, 2)

        print(f"wait: {t0-t_before:.4f}s | SincNet: {t1-t0:.4f}s | GCC: {t2-t1:.4f}s | ResNet: {t3-t2:.4f}s | Conformer: {t4-t3:.4f}s")
        
        return pred
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# 1. 패치분할
def patch_embedding():
    '''이미지를 패치로 분할하는 과정(patch embedding)'''
    # 설정
    image_size = 224
    patch_size = 16
    channels = 3
    embedding_dim = 768

    # 패치수 계싼
    num_patches = (image_size // patch_size) ** 2
    print(f'    이미지 크기 : {image_size} x {image_size}')
    print(f'    패치 크기 : {patch_size} x {patch_size}')
    print(f'    채널수 : {channels}')
    print(f'    패치 수 : {image_size // patch_size} x {image_size // patch_size}')

    # 더미 이미지 생성
    dummy_image = torch.randn(1, channels, image_size, image_size)
    print(f'    더미 이미지 생성')
    print(f'    입력 이미지 shape : {dummy_image.shape}') # [1, 3, 224, 224]

    # 패치분할(Conv2d 사용)
    # Conv2d stride = patch_size 겹치지 않는 패치 추출
    patch_embeded = nn.Conv2d(in_channels=channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)

    # 패치 임베딩 적용
    patches = patch_embeded(dummy_image)
    print(f'\n 패치임베딩 후')
    print(f'    Conv2d 출력 shape : {patches.shape}') # [1, 768, 14, 14]

    # Flatten : (B,D,H,H) -> (B,N,D) (1,196,768)    
    
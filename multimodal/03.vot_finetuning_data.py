import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from torchvision.transforms import (
    RandomResizedCrop,
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 데이터셋 로드
# Food-101 5개 클래스만 선택
selected_classes = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare']
path = r'C:\Users\playdata2\.cache\kagglehub\datasets\dansbecker\food-101\versions\1'
dataset = load_dataset(path, split='train[:1000]')
print(dataset)

# 선택한 클래스만 필터링
def filter_classes(dataset):
    return dataset['label'] in range(len(selected_classes))

dataset = dataset.filter(filter_classes)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"훈련데이터 : {len(dataset['train'])}개")
print(f"테스트 데이터 : {len(dataset['test'])}개")
print(f"클래스 : {selected_classes}")
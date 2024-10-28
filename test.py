import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.metrics import f1_score
import os
from data_load import get_dataloader
from model import get_model

def evaluate_model(model_path, csv_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 데이터 변환 및 데이터 로더 생성
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])
    test_loader = get_dataloader(csv_file, batch_size=16, shuffle=False, transform=transform)

    # 모델 불러오기
    model = get_model(model_type='efficientnet_b1')  # 모델 타입을 지정해주세요
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # 테스트 데이터에 대해 예측 수행
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # F1-Score 계산 및 출력
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    model_path = './models/ef1_model.pth'  # 모델 파일 경로 지정
    csv_file = 'test_obesity_df.csv'  # 테스트 CSV 파일 경로 지정
    evaluate_model(model_path, csv_file)

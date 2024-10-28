import pandas as pd
import os
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ObesityDataset(Dataset):
    def __init__(self, csv_file, transform=None, oversample_classes=None):
        """
        Args:
            csv_file (string): CSV 파일의 경로.
            transform (callable, optional): 이미지에 적용할 변환기.
            oversample_classes (dict, optional): 오버샘플링할 클래스와 배수를 지정하는 딕셔너리. 예: {0: 7, 2: 5}
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.oversample_classes = oversample_classes
        
        # 데이터 증강 설정
        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

        # 오버샘플링 적용
        if self.oversample_classes:
            self.data = self._oversample(self.oversample_classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        images = []

        # 각 이미지 열기 및 크롭하기
        for i in range(1, 14):
            img_path = row[f'img-{str(i).zfill(2)}']
            if pd.isna(img_path):
                continue
            
            img_full_path = os.path.join(img_path)
            if not os.path.exists(img_full_path):
                raise FileNotFoundError(f"Image {img_full_path} not found")

            image = Image.open(img_full_path)

            # 바운딩 박스 좌표 추출
            box = ast.literal_eval(row[f'box-{str(i).zfill(2)}'])
            cropped_image = image.crop(box)

            # 이미지 크기 통일
            cropped_image = cropped_image.resize((224, 224))

            # 변환 적용 (있다면)
            if self.transform:
                cropped_image = self.transform(cropped_image)
            
            # 클래스가 오버샘플링 대상일 경우 데이터 증강 적용
            if self.oversample_classes and row['class'] in self.oversample_classes:
                cropped_image = self.augmentation_transform(cropped_image)

            images.append(transforms.ToTensor()(cropped_image))

        label = row['class']

        if images:
            images = torch.stack(images)
        else:
            images = torch.zeros((1, 3, 224, 224))

        return images, label

    def _oversample(self, oversample_classes):
        """
        오버샘플링을 수행하는 메서드.
        Args:
            oversample_classes (dict): 오버샘플링할 클래스와 배수를 지정하는 딕셔너리. 예: {0: 7, 2: 5}
        """
        frames = [self.data]
        for class_label, multiplier in oversample_classes.items():
            class_data = self.data[self.data['class'] == class_label]
            for _ in range(multiplier - 1):
                frames.append(class_data)
        return pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

def get_dataloader(csv_file, batch_size=8, shuffle=True, transform=None, num_workers=4):
    oversample_classes = {0:7, 2:5}
    dataset = ObesityDataset(csv_file, transform, oversample_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

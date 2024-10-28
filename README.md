# Pet Obesity Detector

## 프로젝트 개요
반려동물의 체중 상태를 **정상**, **저체중**, **비만**으로 분류하는 AI 모델입니다. EfficientNet, ConvNeXt, Vision Transformer (ViT)를 사용하여 성능을 평가하였습니다.

## 사용법

### 1. 환경 설정
Python 환경에서 필요한 라이브러리를 설치하세요. 다음 명령어를 실행합니다:

```bash
pip install -r requirements.txt
```

### 2. 모델 학습
train.py 파일을 실행하여 모델을 학습시킬 수 있습니다. 예시:

```bash
python train.py --model_type efficientnet_b1 --batch_size 16 --num_epochs 30 --learning_rate 0.001
```

### 3. 모델 평가
학습된 모델을 사용하여 테스트 데이터를 평가합니다. 예시:

```bash
python test.py --model_path best_model.pth --csv_file test_obesity_df.csv
```

## 데이터셋
반려동물의 체중 상태를 예측하기 위해 다양한 이미지를 포함한 데이터셋을 사용합니다. 데이터셋은 공공 데이터로, 반려동물의 이미지와 비만도를 라벨링한 데이터를 포함하고 있습니다.

## 주요 라이브러리
프로젝트에서 사용하는 주요 라이브러리는 requirements.txt 파일을 통해 설치할 수 있습니다:

* torch
* torchvision
* pandas
* scikit-learn
* tqdm

## 참고 사항
* 데이터셋은 별도로 다운로드하여 data/ 폴더에 위치시켜야 합니다.
* 학습된 모델은 .pth 형식으로 저장되며, test.py 파일을 통해 평가에 사용됩니다.

## 기여
이 프로젝트에 기여하고 싶다면, 이슈를 생성하거나 풀 리퀘스트를 통해 제안해 주세요.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.
```

이렇게 하면 전체 문서는 하나의 코드 블록이면서, 내부의 명령어들은 ```bash로 표시되어 복사하기 쉬운 형태가 됩니다. 이게 원하시는 형태인가요?

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
https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71520




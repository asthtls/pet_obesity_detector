# pet_obesity_detector
반려동물 비만도 예측


사용법
환경 설정:
Python 환경에서 필요한 라이브러리를 설치합니다. 다음과 같은 명령어를 실행하세요:
sh
코드 복사
pip install -r requirements.txt
모델 학습:
train.py 파일을 실행하여 모델을 학습시킵니다. 예시:
sh
코드 복사
python train.py --model_type efficientnet_b1 --batch_size 16 --num_epochs 30 --learning_rate 0.001
모델 평가:
학습된 모델을 사용하여 테스트 데이터를 평가합니다. 예시:
sh
코드 복사
python test.py --model_path best_model.pth --csv_file test_obesity_df.csv

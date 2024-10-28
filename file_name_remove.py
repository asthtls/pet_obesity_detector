import os
import re

# 이미지와 라벨 파일이 저장된 디렉토리
img_dirs = [
    './dataset/Training/img',
    './dataset/Validation/img'
]

label_dirs = [
    './dataset/Training/label',
    './dataset/Validation/label'
]

# 파일명을 정리하는 함수
def clean_filenames(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # 디렉토리 내의 모든 파일 탐색
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)

        # 정규식을 사용하여 접두사 제거
        # 접두사 예시: TS_A_반려견_1_, TL_A_반려묘_, VS_B_반려견_, VL_B_반려묘_ 등 제거
        # new_filename = re.sub(r'^(TS|TL|VS|VL)_[^_]+_[^_]+_', '', filename)
        
        new_filename = re.sub(r'^\d+_', '', filename)
        
        # 새로운 파일 경로 생성
        new_path = os.path.join(directory, new_filename)

        # 파일명이 변경될 경우에만 실행
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

# 이미지 디렉토리의 파일명 정리
for img_dir in img_dirs:
    clean_filenames(img_dir)

# 라벨 디렉토리의 파일명 정리
for label_dir in label_dirs:
    clean_filenames(label_dir)

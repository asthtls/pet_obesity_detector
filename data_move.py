import os
import shutil


def reorganize_files(source_dirs, img_dest_dir, label_dest_dir):
    """
    소스 디렉토리 목록에서 이미지와 라벨 파일을 대상 디렉토리로 이동하는 함수입니다.

    Parameters:
    - source_dirs (list): 소스 디렉토리의 경로 목록
    - img_dest_dir (str): 이미지 파일을 저장할 대상 디렉토리 경로
    - label_dest_dir (str): 라벨 파일을 저장할 대상 디렉토리 경로
    """
    
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(label_dest_dir, exist_ok=True)

    for dir_path in source_dirs:
        # 디렉토리 내의 모든 파일 탐색
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_lower = file.lower()
                src_path = os.path.join(root, file)

                if file_lower.endswith('.jpg') or file_lower.endswith('.png'):
                    # 이미지 파일 처리
                    new_filename = file  # 원래 파일명 그대로 유지
                    dst_path = os.path.join(img_dest_dir, new_filename)
                    try:
                        shutil.move(src_path, dst_path)
                        print(f"Moved image: {src_path} -> {dst_path}")
                    except Exception as e:
                        print(f"Failed to move image: {src_path}, Error: {e}")
                elif file_lower.endswith('.json'):
                    # 라벨 파일 처리
                    new_filename = file  # 원래 파일명 그대로 유지
                    dst_path = os.path.join(label_dest_dir, new_filename)
                    try:
                        shutil.move(src_path, dst_path)
                        print(f"Moved label: {src_path} -> {dst_path}")
                    except Exception as e:
                        print(f"Failed to move label: {src_path}, Error: {e}")
                else:
                    print(f"Skipped file: {src_path}")

        # 디렉토리가 비었으면 삭제
        if not os.listdir(dir_path):
            try:
                os.rmdir(dir_path)
                print(f"Deleted empty directory: {dir_path}")
            except Exception as e:
                print(f"Failed to delete directory: {dir_path}, Error: {e}")

    
train_source_dirs = [
    './dataset/Training/01.원천데이터/TS_A_반려견_1',
    './dataset/Training/01.원천데이터/TS_A_반려견_2',
    './dataset/Training/01.원천데이터/TS_A_반려견_3',
    './dataset/Training/01.원천데이터/TS_A_반려묘',
    './dataset/Training/01.원천데이터/TS_B_반려견',
    './dataset/Training/01.원천데이터/TS_B_반려묘',
    './dataset/Training/02.라벨링데이터/TL_A_반려견',
    './dataset/Training/02.라벨링데이터/TL_A_반려묘',
    './dataset/Training/02.라벨링데이터/TL_B_반려견',
    './dataset/Training/02.라벨링데이터/TL_B_반려묘',
]

train_img_dest = './dataset/Training/img'
train_label_dest = './dataset/Training/label'

# 파일 재정리 실행 (이미지 및 라벨 파일의 원래 이름 유지)
reorganize_files(train_source_dirs, train_img_dest, train_label_dest)

# 검증 데이터 처리
validation_source_dirs = [
    './dataset/Validation/01.원천데이터/VS_A_반려견',
    './dataset/Validation/01.원천데이터/VS_A_반려묘',
    './dataset/Validation/01.원천데이터/VS_B_반려견',
    './dataset/Validation/01.원천데이터/VS_B_반려묘',
    './dataset/Validation/02.라벨링데이터/VL_A_반려견',
    './dataset/Validation/02.라벨링데이터/VL_A_반려묘',
    './dataset/Validation/02.라벨링데이터/VL_B_반려견',
    './dataset/Validation/02.라벨링데이터/VL_B_반려묘',
]

validation_img_dest = './dataset/Validation/img'
validation_label_dest = './dataset/Validation/label'

# 검증 데이터 파일 재정리 실행 (이미지 및 라벨 파일의 원래 이름 유지)
reorganize_files(validation_source_dirs, validation_img_dest, validation_label_dest)

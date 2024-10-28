from glob import glob
from tqdm import tqdm
import pandas as pd
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def process_dataset(json_dir, img_dir, output_csv):
    ln = 0
    mic = set()
    bfm = "{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"

    # 데이터프레임 컬럼 생성
    columns = ["iid"]
    for i in range(1, 14):
        columns.extend([f"img-{str(i).zfill(2)}", f"box-{str(i).zfill(2)}"])
    columns.append("class")
    df = pd.DataFrame(columns=columns)

    # 메인 JSON 파일 목록 수집
    main_json_files = glob(os.path.join(json_dir, "*.json"), recursive=True)

    if not main_json_files:
        print("No JSON files found. Check dataset path.")
    else:
        print(f"Found {len(main_json_files)} JSON files.")

    for i in tqdm(main_json_files, bar_format=bfm):
        s = 0

        # JSON 파일 열기
        try:
            with open(i, "r", encoding='utf-8') as f:
                d = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file {i}: {e}")
            continue

        try:
            mi = d['metadata']['id']['mission-id']
        except KeyError:
            print(f"Key 'mission-id' not found in {i}")
            continue

        if mi in mic:
            continue

        mic.add(mi)

        # 메타데이터 추출
        try:
            meta = d["metadata"]
            iid = d["annotations"]["image-id"][:-7]  # 이미지 ID에서 마지막 7자를 제거
        except KeyError as e:
            print(f"KeyError for {i}: {e}")
            continue

        temp_list = [iid]

        # 13개의 이미지와 해당 라벨 파일 처리
        for n in range(13):
            img_number = str(n + 1).rjust(2, "0")  # '01', '02', ..., '13'
            img_name = f"{iid}_{img_number}.jpg"

            # 이미지 경로 설정
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                s = 1
                break

            # JSON 파일 경로 설정
            json_name = f"{iid}_{img_number}.json"
            json_path = os.path.join(json_dir, json_name)
            if not os.path.exists(json_path):
                print(f"JSON file not found: {json_name}")
                s = 1
                break

            # JSON 파일 열기 및 바운딩 박스 추출
            try:
                with open(json_path, "r", encoding='utf-8') as f:
                    d = json.load(f)

                points = d["annotations"]["label"]["points"]
                box = points[0] + points[1]  # [x1, y1, x2, y2]
            except (KeyError, FileNotFoundError) as e:
                print(f"Failed to process JSON file {json_path}: {e}")
                s = 1
                break

            # 이미지 경로와 바운딩 박스 정보를 리스트에 추가
            temp_list.append(img_path)
            temp_list.append(box)

        if s:
            continue

        # 클래스 레이블 지정
        try:
            bsc = meta["physical"]["BCS"]
            cls = 2 if bsc >= 6 else 1 if bsc >= 4 else 0
        except KeyError:
            print(f"BCS not found for: {iid}")
            continue

        temp_list.append(cls)

        # 데이터프레임에 행 추가
        df.loc[ln] = temp_list
        ln += 1

    # CSV 파일로 저장
    df.to_csv(output_csv, sep=",", na_rep="NaN", index=False)
    print(f"Saved {ln} records to {output_csv}")


# Training 데이터셋 처리
process_dataset(
    json_dir="./dataset/Training/label",
    img_dir="./dataset/Training/img",
    output_csv="training_obesity_df.csv"
)

# Validation 데이터셋 처리
process_dataset(
    json_dir="./dataset/Validation/label",
    img_dir="./dataset/Validation/img",
    output_csv="validation_obesity_df.csv"
)

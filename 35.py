'''
csv파일 한번 주행당 35 sampling 자동화 코드
'''
import pandas as pd
import os

source_dir = "data"  # 원본 csv 파일 경로
dest_dir = "new"  # 35초 간격으로 처리한 후의 csv 파일 경로

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

name_list = ['choimingi']
# name_list = ['leeseunglee']
# name_list = ['jojeongdeok','leeyunguel','leegahyeon','huhongjune','jeongyubin','leegihun','leejaeho','leekanghyuk','simboseok']
# name_list = ['jojeongdeok','leeyunguel','leegahyeon', 'jeongyubin','simboseok']
course_list = ['A', 'B', 'C']
round_list = ['1', '2', '3', '4']

for name in name_list:
    for course in course_list:
        for round in round_list:
            file_path = f"{source_dir}/{name}/{course}/{round}/Accelerometer.csv"

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                start_time = 0
                end_time = 35

                # 필요한 하위 디렉토리 생성
                dest_subdir = os.path.join(dest_dir, name, course, round)
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir)

                while end_time <= df['seconds_elapsed'].max():
                    filtered_df = df[(df['seconds_elapsed'] >= start_time) & (df['seconds_elapsed'] <= end_time)]
                    file_name = os.path.join(dest_subdir, f"Filtered_Accelerometer_{start_time}-{end_time}.csv")
                    filtered_df.to_csv(file_name, index=False)
                    start_time = end_time
                    end_time += 35

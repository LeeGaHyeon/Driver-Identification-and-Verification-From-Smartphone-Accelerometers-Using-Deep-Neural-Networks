'''
make csv
'''
import os
import csv
import glob

# Label 및 Course에 대한 매핑 딕셔너리
label_mapping = {
    'jojeongdeok':0,
    'leeyunguel':1,
    'leegahyeon':2,
    'huhongjune':3,
    'jeongyubin':4,
    'leegihun':5,
    'leejaeho':6,
    'leekanghyuk':7,
    'simboseok':8,
    'leeseunglee':9,
    'choimingi':10,
}

course_mapping = {
    "A": 0,
    "B": 1,
    "C": 2
}

round_mapping = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3
}

name_lt = ['jojeongdeok','leeyunguel','leegahyeon','huhongjune','jeongyubin','leegihun','leejaeho','leekanghyuk','simboseok', 'choimingi']
# name_lt = ["jojeongdeok", 'leeyunguel', 'leegahyeon', 'jeongyubin','simboseok']
course = ["A", "B", "C"]
round_lt = ["1", "2", "3", "4"]

# CSV 파일 생성 및 열 이름 설정
csv_filename = "11_longitudinal_spectrogram.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["label", "course", "round", "file"])
    frame = 0
    for name in name_lt:

        for c in course:
            for r in round_lt:
                path = f'./longitudinal_spectrogram/{name}_spectrogram/{c}/{r}'
                file_list = os.listdir(path)  # 파일명

                for f in file_list:
                    frame += 1
                    file = os.path.join(path, f)

                    label = label_mapping.get(name, -1)  # Label 매핑
                    course_name = course_mapping.get(c, -1)  # Course 매핑
                    round = round_mapping.get(r, -1)
                    # print(label, course_name, file)
                    # CSV 파일에 데이터 쓰기
                    writer.writerow([label, course_name, r, file])
                print(f'{name}_{c} --> done')

print(f'CSV 파일 "{csv_filename}"이 생성되었습니다.')

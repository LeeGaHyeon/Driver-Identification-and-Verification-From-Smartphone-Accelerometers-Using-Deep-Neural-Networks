import pandas as pd
import numpy as np
import matplotlib.cm as cm
from scipy.signal import spectrogram
import cv2
import os

source_dir = "new"  # 소스 디렉토리 경로를 수정하세요.
dest_dir = "longitudinal_spectrogram"  # 대상 디렉토리 경로를 설정하세요.

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

name_list = ['choimingi']
# name_list = ['leeseunglee']
# name_list = ['jojeongdeok','leeyunguel','leegahyeon','huhongjune','jeongyubin','leegihun','leejaeho','leekanghyuk','simboseok']
# name_list = ['jojeongdeok','leeyunguel','leegahyeon', 'jeongyubin','simboseok']
course_list = ['A','B','C']
round_list = ['1','2','3','4']

for name in name_list:
    for course in course_list:
        for round in round_list:
            source_path = os.path.join(source_dir, name, course, round)
            dest_path = os.path.join(dest_dir, f"{name}_spectrogram", course, round)

            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            n = 0

            for file_name in os.listdir(source_path):
                if file_name.endswith(".csv"):
                    n += 1
                    # CSV 파일 로드
                    data = pd.read_csv(os.path.join(source_path, file_name))
                    time = data['seconds_elapsed'] - data['seconds_elapsed'].iloc[0]

                    longitudinal_acceleration = data['x']

                    f, t, Sxx = spectrogram(longitudinal_acceleration, fs=5, nperseg=224, noverlap=112)
                    Sxx = np.log(Sxx)
                    Sxx_normalized = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())

                    colored_spectrogram = cm.viridis(Sxx_normalized)
                    colored_spectrogram_bgr = cv2.cvtColor((colored_spectrogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                    filename = os.path.splitext(file_name)[0]
                    npy_save_path = os.path.join(dest_path, f"spectrogram_{n}.npy")
                    np.save(npy_save_path, np.array(colored_spectrogram_bgr))

                    n += 1
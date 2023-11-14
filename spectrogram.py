# '''
# conversion from 1d signals to 2d signals (1초, 0stride)
# '''
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import spectrogram
# import cv2
# import os
# import matplotlib.cm as cm
#
# # Define the source and destination directories
# source_dir = "data"  # Modify this to point to your source directory
# dest_dir = "longitudinal_spectrogram"  # Modify this to set the destination directory
#
# # Create the destination directory if it doesn't exist
# if not os.path.exists(dest_dir):
#     os.makedirs(dest_dir)
#
# # Define the lists
# name_list = ['jojeongdeok', 'leeyunguel']
# course_list = ['A', 'B', 'C']
# round_list = ['1','2','3','4']
#
# # Iterate over the lists to create the spectrograms
# for name in name_list:
#     for course in course_list:
#         for round in round_list:
#             # Create the directory path for the source CSV files
#             source_path = os.path.join(source_dir, name, course, round)
#
#             # Create the directory path for the destination images
#             dest_path = os.path.join(dest_dir, f"{name}_spectrogram", course, round)
#
#             # Create the directory if it doesn't exist
#             if not os.path.exists(dest_path):
#                 os.makedirs(dest_path)
#
#             # Iterate over CSV files in the source directory
#             for file_name in os.listdir(source_path):
#                 if file_name.endswith(".csv"):
#                     # Load the CSV file
#                     data = pd.read_csv(os.path.join(source_path, file_name))
#                     time = data['seconds_elapsed'] - data['seconds_elapsed'].iloc[0]
#
#                     longitudinal_acceleration = data['x']
#                     transversal_acceleration = data['y']
#                     angular_velocity = data['z']
#
#                     sampling_frequency = 1 / (time[1] - time[0])
#
#                     # os.makedirs("spectrograms", exist_ok=True)
#
#                     for start_time in range(0, len(time), int(sampling_frequency)):
#                         end_time = start_time + int(sampling_frequency)
#                         if end_time <= len(time):
#                             f, t, Sxx = spectrogram(angular_velocity[start_time:end_time], fs=sampling_frequency, nperseg=64)
#                             Sxx = np.log(Sxx)
#                             desired_size = (224, 224)
#                             Sxx_resized = cv2.resize(Sxx, desired_size)
#                             Sxx_normalized = (Sxx_resized - Sxx_resized.min()) / (Sxx_resized.max() - Sxx_resized.min())
#                             colored_spectrogram = cm.viridis(Sxx_normalized)
#                             colored_spectrogram_bgr = cv2.cvtColor((colored_spectrogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
#
#                             filename = os.path.splitext(file_name)[0]  # Remove the '.csv' extension
#                             # save_path = os.path.join(dest_path, f"{filename}_spectrogram_{start_time}_{end_time}.png")
#                             # cv2.imwrite(save_path, colored_spectrogram_bgr)
#                             npy_save_path = os.path.join(dest_path, f"spectrogram_{start_time}_{end_time}.npy")
#                             np.save(npy_save_path, np.array(colored_spectrogram_bgr))

'''
1초 간격으로 스펙트로그램을 생성하지 않고, 60초를 기준으로 1초 간격으로 스트라이드하여 겹치는 스펙트로그램을 생성하도록 수정된 코드
# '''
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import spectrogram
# import cv2
# import os
# import matplotlib.cm as cm
# from sklearn.decomposition import PCA
#
# source_dir = "data"  # 소스 디렉토리 경로
# ############수정#################
# dest_dir = "longitudinal_spectrogram"  # 대상 디렉토리 경로
# if not os.path.exists(dest_dir):
#     os.makedirs(dest_dir)
#
# # 목록 정의
# name_list = ['jojeongdeok', 'leeyunguel']
# course_list = ['A', 'B', 'C']
# round_list = ['1', '2', '3', '4']
#
# for name in name_list:
#     for course in course_list:
#         for round in round_list:
#             source_path = os.path.join(source_dir, name, course, round)
#
#             dest_path = os.path.join(dest_dir, f"{name}_spectrogram", course, round)
#
#             if not os.path.exists(dest_path):
#                 os.makedirs(dest_path)
#
#             for file_name in os.listdir(source_path):
#                 if file_name.endswith(".csv"):
#                     # CSV 파일 로드
#                     data = pd.read_csv(os.path.join(source_path, file_name))
#                     time = data['seconds_elapsed'] - data['seconds_elapsed'].iloc[0]
#
#                     longitudinal_acceleration = data['x']
#
#                     # sampling_frequency = 1 / (time[1] - time[0])
#
#                     for start_time in range(0, len(time) - 447, 224):
#                         end_time = start_time + 448
#                         ############수정#################
#                         f, t, Sxx = spectrogram(longitudinal_acceleration, fs=5, nperseg=224, noverlap=112) # nperseg: 각 세그먼트의 길이
#                         Sxx = np.log(Sxx)
#                         Sxx_normalized = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())
#
#                         colored_spectrogram = cm.viridis(Sxx_normalized)
#                         colored_spectrogram_bgr = cv2.cvtColor((colored_spectrogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
#
#                         filename = os.path.splitext(file_name)[0]
#                         npy_save_path = os.path.join(dest_path, f"spectrogram_{start_time}_{end_time}.npy")
#                         np.save(npy_save_path, np.array(colored_spectrogram_bgr))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import cv2
import os
import matplotlib.cm as cm
from sklearn.decomposition import PCA

# 소스 및 대상 디렉토리 정의
source_dir = "new"  # 소스 디렉토리 경로를 수정하세요.
############수정#################
dest_dir = "longitudinal_spectrogram"  # 대상 디렉토리 경로를 설정하세요.

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 목록 정의
# name_list = ['jojeongdeok', 'leeyunguel', 'leegahyeon']
name_list = ['choimingi']
course_list = ['A', 'B', 'C']
round_list = ['1', '2', '3', '4']

for name in name_list:
    for course in course_list:
        for round in round_list:
            source_path = os.path.join(source_dir, name, course, round)

            dest_path = os.path.join(dest_dir, f"{name}_spectrogram", course, round)

            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            for file_name in os.listdir(source_path):
                if file_name.endswith("Accelerometer.csv"):
                    # CSV 파일 로드
                    data = pd.read_csv(os.path.join(source_path, file_name))
                    time = data['seconds_elapsed'] - data['seconds_elapsed'].iloc[0]

                    longitudinal_acceleration = data['x']

                    # sampling_frequency = 1 / (time[1] - time[0])

                    for start_time in range(0, len(time) - 447, 224):
                        end_time = start_time + 448
                        ############수정#################
                        f, t, Sxx = spectrogram(longitudinal_acceleration, fs=5, nperseg=224, noverlap=112) # nperseg: 각 세그먼트의 길이
                        Sxx = np.log(Sxx)
                        Sxx_normalized = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())

                        colored_spectrogram = cm.viridis(Sxx_normalized)
                        colored_spectrogram_bgr = cv2.cvtColor((colored_spectrogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                        filename = os.path.splitext(file_name)[0]
                        npy_save_path = os.path.join(dest_path, f"spectrogram_{start_time}_{end_time}.npy")
                        np.save(npy_save_path, np.array(colored_spectrogram_bgr))



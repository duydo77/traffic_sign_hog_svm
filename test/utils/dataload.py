import os
import cv2
from cv2 import imread
import pandas as pd

data_label = ['14', '17', '38', '39', '40']

def load_data(input_size = (64,64), data_path =  './image/GTSRB/Final_Training/Images'):

    pixels = []
    labels = []
    # Loop qua các thư mục trong thư mục Images
    for dir in os.listdir(data_path):
        if dir == '.DS_Store':
            continue

        # Đọc file csv để lấy thông tin về ảnh
        class_dir = os.path.join(data_path, dir)
        info_file = pd.read_csv(os.path.join(class_dir, "GT-" + dir + '.csv'), sep=';')

        print(info_file)
        # Lăp trong file
    #     for row in info_file.iterrows():
    #         # Đọc ảnh
    #         pixel = imread(os.path.join(class_dir, row[1].Filename))
    #         # Trích phần ROI theo thông tin trong file csv
    #         pixel = pixel[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
    #         # Resize về kích cỡ chuẩn
    #         img = cv2.resize(pixel, input_size)

    #         # Thêm vào list dữ liệu
    #         pixels.append(img)

    #         # Thêm nhãn cho ảnh
    #         labels.append(row[1].ClassId)

    # return pixels, labels

if __name__ == '__main__':
    a = load_data()
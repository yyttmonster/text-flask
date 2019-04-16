import os
import csv
import re
import shutil
import cv2
import numpy as np


def alter_format():
    gt_path_name = "D:/Data/DATA/ICDAR2013/icdar13-Training-GT"
    gt_list = os.listdir(gt_path_name)
    for file in gt_list:
        file_path = gt_path_name + "/" + file
        strs = ""
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = re.split(" ", line)
                label = eval(line[-1])
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
                x1, y1, x3, y3 = list(map(int, line[:4]))
                x2 = x3
                y2 = y1
                x4 = x1
                y4 = y3
                strs += str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," \
                        + str(x3) + "," + str(y3) + "," + str(x4) + "," + str(y4) + "," \
                        + label + "\n"
                # print(file_path, ":", strs)
        with open(file_path, 'w', encoding='UTF-8') as f:
            f.write(strs)
        # print(file_path,":",strs)


def delete_gt():
    img_path_name = 'D:/Data/HUST-TR400/img'
    gt_path_name = "D:/Data/HUST-TR400/txt"
    img_list = os.listdir(img_path_name)
    gt_list = os.listdir(gt_path_name)
    i = 0
    for file in gt_list:
        # image_name = re.sub('IMG_', '', file)
        image_name = re.sub('txt', 'jpg', file)
        print(image_name)
        if not image_name in img_list:
            os.remove(gt_path_name + '\\' + file)
        # print(os.path.pathname(file))
        # os.rename(img_path_name+file, new_name)


def data_rename():
    img_path_name = 'D:/Data/DATA/MLT/img/'
    # middle = "D:/Data/DATA/icdar2015/middle/"
    # gt_path_name = "D:\Data\icdar2015\ch4_test_images_gt"
    i = 3048
    for file in os.listdir(img_path_name):
        # print(file)
        new_name = "img_" + str(i) + ".jpg"
        i += 1
        print(new_name, img_path_name + file)
        os.rename(img_path_name + file, img_path_name + new_name)


def select_gt():
    img_path_name = 'D:/Data/DATA/MLT/img/'
    gt_path_name = "D:/Data//MLT/MLT_gt/"
    new_gt_path = "D:/Data/DATA/MLT/gt/"
    for file in os.listdir(img_path_name):
        gt_name = "gt_" + re.sub('jpg', 'txt', file)
        # print(gt_name)
        shutil.copyfile(gt_path_name + gt_name, new_gt_path + gt_name)


def is_latin():
    gt_path = "D:/Data/DATA/MLT/ch8_training_images_7_gt/"
    gt_list = os.listdir(gt_path)
    for file in gt_list:
        strs = ""
        with open(gt_path + file, "r", encoding='UTF-8') as f:
            for line in f:
                line = re.split(",", line)
                if not line[8] == "Latin":
                    line[9] = "###\n"
                if len(line) > 10:
                    for i in range(10, len(line)):
                        line[9] += line[i]
                    # print(line[9])
                strs += line[0] + "," + line[1] + "," + line[2] + "," + line[3] + "," + line[4] + "," + line[5] + "," + \
                        line[6] + "," + line[7] + "," + line[9]
            # print(strs)
        with open(gt_path + file, "w", encoding='UTF-8') as f:
            f.write(strs)

        # print(type(line))
        # print(line[-2])


def divide_words_img():
    img_path = r"D:/Data/DATA/dataset/img/"
    gt_path = r"D:/Data/DATA/dataset/img_gt/"
    words_path = r"D:/Data/DATA/dataset/words/"
    img_list = os.listdir(img_path)
    gt_list = os.listdir(gt_path)
    for img_name in img_list:
        gt_name = re.sub("img", "gt_img", (re.sub("jpg", "txt", img_name)))
        print(gt_name)
        if gt_name not in gt_list:
            print("Couldn't find the txt file of " + img_name)
            continue
        with open(gt_path + gt_name, "r", encoding="UTF-8") as f:
            word_count = 1
            img = cv2.imread(img_path + img_name)
            for line in f:
                line = re.sub(r"\ufeff", "", line)
                line = re.split(r",|\n", line)
                if len(line) < 8:
                    continue
                x1, y1, x2, y2, x3, y3, x4, y4 = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(
                    line[4]), int(line[5]), int(line[6]), int(line[7])
                x_min = min(x1, x2, x3, x4)
                x_max = max(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                y_max = max(y1, y2, y3, y4)
                word_background = np.zeros((np.int32(y_max - y_min), np.int32(x_max - x_min)), dtype=np.int32)
                poly_area = np.array([[x1 - x_min, y1 - y_min], [x2 - x_min, y2 - y_min], [x3 - x_min, y3 - y_min],
                                      [x4 - x_min, y4 - y_min]])
                cv2.fillPoly(word_background, np.int32([poly_area]), 1)
                word_area = np.copy(img[y_min:y_max, x_min:x_max])
                try:
                    word_area[:, :, 0] *= np.uint8(word_background)
                    word_area[:, :, 1] *= np.uint8(word_background)
                    word_area[:, :, 2] *= np.uint8(word_background)
                    cv2.imwrite(filename=words_path + re.sub(".jpg", "_word_" + str(word_count) + ".jpg", img_name),
                                img=word_area)
                    word_count += 1
                except Exception as e:
                    print("\033[0;31m", gt_name, "\033[0m", e)
                    print("\033[0;31m",
                          "Shape don't match! Maybe some negative numbers exist! The type must be 'uint'!", "\033[0m")
                # cv2.imshow("img", word_area)
                # cv2.waitKey(0)


def alter():
    gt_path_name = "D:/Data/DATA/ICDAR2013/icdar13_Test_GT"
    gt_list = os.listdir(gt_path_name)
    for file in gt_list:
        file_path = gt_path_name + "/" + file
        strs = ""
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = re.split(",", line)
                if len(line) < 7:
                    continue
                label = line[-1]
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
                x1, y1, x2, y2, x4, y4, x3, y3 = list(map(int, line[:8]))
                strs += str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," \
                        + str(x3) + "," + str(y3) + "," + str(x4) + "," + str(y4) + "," \
                        + label
                print(file_path, ":", strs)
        with open(file_path, 'w', encoding='UTF-8') as f:
            f.write(strs)


def add_txt():
    gt_path_name = "D:/Data/DATA/USTB/training/USTB_train_txt/"
    gt_list = os.listdir(gt_path_name)
    for gt_name in gt_list:
        strs = ""
        with open(gt_path_name + gt_name, "r", encoding="UTF-8") as f:
            for line in f:
                line = re.sub("\n", ",Text\n", line)
                strs += line
        print(gt_name)
        with open(gt_path_name + gt_name, "w", encoding='UTF-8') as f:
            f.write(strs)


def is_exist_space():
    gt_path_name = "D:/Data/DATA/dataset/img_gt/"
    gt_list = os.listdir(gt_path_name)
    for gt_name in gt_list:
        with open(gt_path_name + gt_name, "r", encoding="UTF-8") as f:
            for line in f:
                line = re.split(",", line)
                if len(line) != 9:
                    print(gt_name, line)


def is_exist_symbol():
    gt_path_name = "D:/Data/DATA/dataset/img_gt/"
    gt_list = os.listdir(gt_path_name)
    for gt_name in gt_list:
        strs = ""
        with open(gt_path_name + gt_name, "r", encoding="UTF-8") as f:
            for line in f:
                if re.match("\ufeff", line):
                    print(gt_name)
        #         line = re.sub("\ufeff","",line)
        #         strs += line
        #     print(strs)
        # with open(gt_path_name + gt_name, "w", encoding='UTF-8') as f:
        #     f.write(strs)


def main():
    divide_words_img()


main()

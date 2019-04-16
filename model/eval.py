import json
import os
import re
import numpy as np
from shapely.geometry import Polygon, MultiPoint


def compute_iou(list_1, list_2):
    if len(list_1) != 8 and len(list_2) != 8:
        print("list length must be eight corresponding to four coordinates!")
        return 0
    a = np.array(list_1).reshape(4, 2)
    b = np.array(list_2).reshape(4, 2)
    poly_1 = Polygon(a).convex_hull
    poly_2 = Polygon(b).convex_hull
    union = np.concatenate((a, b))
    if not poly_1.intersects(poly_2):
        return 0
    else:
        try:
            intersection_area = poly_1.intersection(poly_2).area
            if intersection_area == 0:
                return 0
            union_area = MultiPoint(union).convex_hull.area
            return float(intersection_area) / union_area
        except Exception:
            print(Exception)
            return 0


def eval(gt_base_path, json_base_path):
    # word_real_num = 0
    # word_detect_num = 0
    # word_right_num = 0
    recall = 0
    precision = 0
    gt_file_list = os.listdir(gt_base_path)
    with open(json_base_path, "r") as fr:
        json_dict = json.load(fr)
        for file in gt_file_list:
            word_real_num = 0
            word_right_num = 0
            img_name = re.sub("gt_", "", file)
            img_name = re.sub(".txt", ".jpg", img_name)
            current_annotations = json_dict[img_name]["annotations"]
            word_detect_num = json_dict[img_name]["box_num"]
            with open(gt_base_path + file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = re.sub('\ufeff',"",line)
                    elements = re.split(",|\n", line)
                    if len(elements) < 9:
                        print("\033[0;31m", "error text line", "\033[0m", elements)
                        continue
                    content = elements[8]
                    # need not detect
                    if content == "###":
                        continue
                    # left bottom is beginning
                    word_real_num += 1
                    box_gt = [int(elements[6]), int(elements[7]), int(elements[0]), int(elements[1]), int(elements[2]),
                              int(elements[3]), int(elements[4]), int(elements[5])]
                    for element in current_annotations:
                        if str.lower(element["text"]) == str.lower(content):
                            iou = compute_iou(box_gt, element["bbox"])
                            if iou > 0.5:
                                word_right_num += 1
                                continue
                print(file,word_real_num, word_detect_num, word_right_num)

                if word_right_num == 0:
                    if word_detect_num == 0:
                        precision += 1
                    if word_real_num == 0:

                        recall += 1

                    continue

                recall += float(word_right_num) / word_real_num
                precision += float(word_right_num) / word_detect_num

                # break
    # print(word_real_num, word_detect_num, word_right_num)
    print("recall:", recall/1358)
    print("precision:", precision/1358)
    print("F1_score:", 2 * precision * recall / (precision + recall)/1358)


gt_base_path = "test_images/output/"
json_base_path = "scenetext_result/scenetext_result.json"
eval(gt_base_path, json_base_path)

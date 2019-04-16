import glob
import os
import csv
import cv2
import numpy as np


def get_images(file_path):
    '''
    :param file_path: path of training data
    :return:
    '''
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(file_path, '*.{}'.format(ext))))
    return files


def load_annoataion(path_annotation):
    '''
    load annotation from the text file
    :param path_annotation: path of annotation
    :return:
    '''
    text_polys = []
    text_tags = []
    # if not os.path.exists(path_annotation):
    #     return np.array(text_polys, dtype=np.float32)
    with open(path_annotation, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            # read parameter and cast to float
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def generate_rbox(image, polys, tags):
    for i in range(polys.shape[0]):
        if not tags[i]:
            cv2.line(image, tuple(polys[i][0]), tuple(polys[i][1]), color=(0, 0, 255))
            cv2.line(image, tuple(polys[i][1]), tuple(polys[i][2]), color=(0, 0, 255))
            cv2.line(image, tuple(polys[i][2]), tuple(polys[i][3]), color=(0, 0, 255))
            cv2.line(image, tuple(polys[i][3]), tuple(polys[i][0]), color=(0, 0, 255))


def text_pixel_extract(image, polys, tags):
    # 得到灰度图 H*W
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 得到灰度直方图
    hist_array = cv2.calcHist([im_gray], [0], None, [256], [0.0, 256.0])
    # 针对每一个文本框内的文本进行操作
    for text_index in range(polys.shape[0]):
        # 小文本不必提取，自行忽略
        if tags[text_index]:
            continue
        # 得到能完全包含整个文本框的水平矩形
        poly_min_x = np.int32(np.min(polys[text_index][:, 0]))
        poly_max_x = np.int32(np.max(polys[text_index][:, 0]))
        poly_min_y = np.int32(np.min(polys[text_index][:, 1]))
        poly_max_y = np.int32(np.max(polys[text_index][:, 1]))
        text_area = np.zeros((np.int32(poly_max_y - poly_min_y), np.int32(poly_max_x - poly_min_x)))
        # 将文本区域填充为1
        new_poly = polys[text_index][:]
        new_poly[0, 0] = new_poly[0, 0] - poly_min_x
        new_poly[1, 0] = new_poly[1, 0] - poly_min_x
        new_poly[2, 0] = new_poly[2, 0] - poly_min_x
        new_poly[3, 0] = new_poly[3, 0] - poly_min_x

        new_poly[0, 1] = new_poly[0, 1] - poly_min_y
        new_poly[1, 1] = new_poly[1, 1] - poly_min_y
        new_poly[2, 1] = new_poly[2, 1] - poly_min_y
        new_poly[3, 1] = new_poly[3, 1] - poly_min_y

        print(new_poly)
        text_area = cv2.fillPoly(text_area, np.int32([new_poly]), 1)
        # 将01图与原灰度图区域对应元素相乘，得到真正文本四边形区域的灰度图（矩形非文本区域为0）
        text_area = im_gray[poly_min_y:poly_max_y, poly_min_x:poly_max_x]
        text_area = text_area.astype(np.uint8)
        print(text_area[0][0],type(im_gray),type(text_area),type(im_gray[0][0]),type(text_area[0][0]))
        threshold, result = cv2.threshold(text_area, 0, 255, cv2.THRESH_OTSU)

        print(result[5])
        cv2.imshow('name', result)
        cv2.waitKey(0)
        return result

        # # 这是基于显著区域提取LC法的提取操作
        # dist= {}
        # ret2, th2 = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # image_gray_copy = np.zeros((image.shape[0], image.shape[1]))
        # for gray in range(256):
        #     value = 0.0
        #     for k in range(256):
        #         value += hist_array[k][0] * abs(gray - k)
        #     dist[gray] = value
        # for i in range(im_gray.shape[1]):
        #     for j in range(im_gray.shape[0]):
        #         temp = im_gray[j][i]
        #         image_gray_copy[j][i] = dist[temp]
        # image_gray_copy = (image_gray_copy - np.min(image_gray_copy)) / (np.max(image_gray_copy) - np.min(image_gray_copy))
        # cv2.imshow('gray',image_gray_copy)
        # cv2.waitKey(0)


def main():
    path_train_data = 'data/ch4_training_images'
    path_ground_truth = 'data/ch4_ground_truth/'
    image_list = np.array(get_images(path_train_data))
    image_number = np.arange(0, image_list.shape[0])
    for image_index in image_number:
        image_name = image_list[image_index]
        image = cv2.imread(image_name)

        # print(type(image))
        annotation_name = image_name.replace(os.path.basename(image_name).split('.')[1], 'txt')
        annotation_name = annotation_name.replace('images\\', 'localization\gt_')
        text_polys, text_tags = load_annoataion(annotation_name)

        if image_index == 1:
            text_pixel_extract(image, text_polys, text_tags)

        generate_rbox(image, text_polys, text_tags)
        name = image_name.split('\\')[1]
        print(name, '::', image_index)
        print(image.shape)
        cv2.imwrite(path_ground_truth + name, image)


if __name__ == '__main__':
    main()

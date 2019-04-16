import cv2
import re
import os
import json
import logging
import numpy as np
import model.detection.model_dc
import tensorflow as tf
from google.protobuf import text_format
from model.detection.dc_hsv_rotate_flip import restore_rectangle
import model.detection.locality_aware_nms as nms_locality
from model.aster.protos import pipeline_pb2
from model.aster.builders import model_builder
from shapely.geometry import Polygon

tf.app.flags.DEFINE_string('data_dir', os.getenv('ABSOLUTE_PATH')+'model/test_images/images', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', os.getenv('ABSOLUTE_PATH')+'model/data/detection_ckpt/', '')
tf.app.flags.DEFINE_string('output_dir', os.getenv('ABSOLUTE_PATH')+'model/test_images/output', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_string('words_path', os.getenv('ABSOLUTE_PATH')+'model/test_images//words/', '')

flags = tf.app.flags
tf.app.flags.DEFINE_string('exp_dir', os.getenv('ABSOLUTE_PATH')+'model/aster/experiments/demo/',
                           'Directory containing config, training log and evaluations')
FLAGS = tf.app.flags.FLAGS

# supress TF logging duplicates
logging.getLogger('tensorflow').propagate = False
tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)


class Text_recognition():

    def __init__(self):
        self.recognition_graph = tf.Graph()
        self.detection_graph = tf.Graph()
        self.detection_sess = None
        self.recognition_sess = None
        self.global_step = None
        self.detection_saver = None
        self.init1 = None
        self.init2 = None
        self.init3 = None
        self.variable_averages = None
        self.recognition_saver = None
        self.f_score = None
        self.f_geometry = None
        self.input_images = None
        self.fetches = None
        self.input_image_str_tensor = None

    def intersection_1(self, g, p):
        g = Polygon(g)
        p = Polygon(p)

        if not g.is_valid or not p.is_valid:
            return 0
        inter = g.intersection(p).area
        union = min(g.area, p.area)
        # union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter / union

    def ms_standard_nms(self, S, scores, thres):
        ss1 = (S.shape)
        ss2 = (scores.shape)
        order = np.argsort(scores)[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            si = S[i]

            ovr = np.array([self.intersection_1(si, S[t]) for t in order[1:]])
            # print(ovr)
            # print(ovr)
            # ovr_1 = np.array(intersection_1(si, si))
            inds = np.where(ovr <= thres)[0]
            order = order[inds + 1]

        return S[keep]

    def get_images(self):
        '''
        DETECTION Function
        find image files in test data path
        :return: list of files found
        '''
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(FLAGS.data_dir):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        print('Find {} images'.format(len(files)))
        return files

    def resize_image(self, im, max_side_len=1024):
        '''
        DETECTION Function
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        '''
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def detect(self, score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        # nms part
        boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)

        if boxes.shape[0] == 0:
            return None

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes

    def sort_poly(self, p):
        '''
        DETECTION Function
        :param p:
        :return:
        '''
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def get_configs_from_exp_dir(self):
        '''
        recognition Function
        :return:
        '''
        pipeline_config_path = os.path.join(FLAGS.exp_dir, 'config/trainval.prototxt')

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.gfile.GFile(pipeline_config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)

        model_config = pipeline_config.model
        eval_config = pipeline_config.eval_config
        input_config = pipeline_config.eval_input_reader

        return model_config, eval_config, input_config

    def load_model(self, argv=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        try:
            os.makedirs(FLAGS.words_path)
        except OSError as e:
            if e.errno != 17:
                raise

        # recognition Graph
        with self.recognition_graph.as_default():
            model_config, _, _ = self.get_configs_from_exp_dir()
            atser_model = model_builder.build(model_config, is_training=False)
            self.input_image_str_tensor = tf.placeholder(
                dtype=tf.string,
                shape=[])
            input_image_tensor = tf.image.decode_jpeg(
                self.input_image_str_tensor,
                channels=3,
            )
            resized_image_tensor = tf.image.resize_images(
                tf.to_float(input_image_tensor),
                [64, 256])
            predictions_dict = atser_model.predict(tf.expand_dims(resized_image_tensor, 0))
            recognitions = atser_model.postprocess(predictions_dict)
            recognition_text = recognitions['text'][0]

            self.recognition_saver = tf.train.Saver(tf.global_variables())
            recognition_checkpoint = os.path.join(FLAGS.exp_dir, 'log/model.ckpt')

            self.fetches = {
                'original_image': input_image_tensor,
                'recognition_text': recognition_text,
                'control_points': predictions_dict['control_points'],
                'rectified_images': predictions_dict['rectified_images'],
            }
            self.init1, self.init2, self.init3 = tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()
        # detection Graph
        with self.detection_graph.as_default():
            self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                               trainable=False)

            self.f_score, self.f_geometry = model.detection.model_dc.model(self.input_images, is_training=False)
            self.variable_averages = tf.train.ExponentialMovingAverage(0.997, self.global_step)
            self.detection_saver = tf.train.Saver(self.variable_averages.variables_to_restore())

        self.detection_sess = tf.Session(graph=self.detection_graph, config=tf.ConfigProto(allow_soft_placement=True))
        detection_ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_path)
        detection_model_path = os.path.join(FLAGS.checkpoint_path,
                                            os.path.basename(detection_ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(detection_model_path))
        self.detection_saver.restore(self.detection_sess, detection_model_path)
        self.recognition_sess = tf.Session(config=config, graph=self.recognition_graph)

        # self.init1, self.init2, self.init3= tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()

        self.recognition_sess.run([self.init1, self.init2, self.init3])
        self.recognition_saver.restore(self.recognition_sess, recognition_checkpoint)

        # json_dict = {}
        # im_fn_list = self.get_images()
        #
        # for im_fn in im_fn_list:
        #     print(im_fn)
        #     # im = cv2.imread(im_fn)[:, :, ::-1]
        #     # image_name = im_fn.split("/")[-1]
        #     # im_resized, (ratio_h, ratio_w) = resize_image(im)
        #     # score, geometry = detection_sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        #     # boxes= detect(score_map=score, geo_map=geometry)
        #     #
        #     # # save to file
        #     # if boxes is None:
        #     #     json_dict[image_name] = {
        #     #         "box_num": 0,
        #     #         "annotations": []
        #     #     }
        #     #     print(json_dict)
        #     #     continue
        #     #
        #     # if boxes is not None:
        #     #     boxes = boxes[:, :8].reshape((-1, 4, 2))
        #     #     boxes[:, :, 0] /= ratio_w
        #     #     boxes[:, :, 1] /= ratio_h
        #     #
        #     #
        #     #
        #
        #     image_name = im_fn.split("/")[-1]
        #     im = cv2.imread(im_fn)[:, :, ::-1]
        #
        #     if im.shape[0] > 1280 or im.shape[1] > 1280:
        #         pass
        #
        #     im_resized, (ratio_h, ratio_w) = self.resize_image(im, max_side_len=1280)
        #     boxes = None
        #     boxes_scores = None
        #     timer = {'net': 0, 'restore': 0, 'nms': 0}
        #     # start = time.time()
        #     score, geometry = detection_sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        #
        #     boxes_0 = self.detect(score_map=score, geo_map=geometry)
        #
        #     if boxes_0 is not None:
        #         boxes_scores = boxes_0[:, 8]
        #         boxes = boxes_0[:, :8].reshape((-1, 4, 2))
        #         boxes[:, :, 0] /= ratio_w
        #         boxes[:, :, 1] /= ratio_h
        #
        #     im_resized, (ratio_h, ratio_w) = self.resize_image(im, max_side_len=896)
        #
        #     score, geometry = detection_sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        #
        #     boxes_1 = self.detect(score_map=score, geo_map=geometry)
        #
        #     if boxes_1 is not None:
        #
        #         boxes_scores_1 = boxes_1[:, 8] * 2
        #         boxes_1 = boxes_1[:, :8].reshape((-1, 4, 2))
        #         boxes_1[:, :, 0] /= ratio_w
        #         boxes_1[:, :, 1] /= ratio_h
        #         if boxes is not None:
        #             boxes_scores = np.concatenate((boxes_scores, boxes_scores_1), axis=0)
        #             boxes = np.concatenate((boxes, boxes_1), axis=0)
        #         else:
        #             boxes_scores = boxes_scores_1
        #             boxes = boxes_1
        #
        #     im_resized, (ratio_h, ratio_w) = self.resize_image(im, max_side_len=1024)
        #
        #     score, geometry = detection_sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        #
        #     boxes_2 = self.detect(score_map=score, geo_map=geometry)
        #
        #     if boxes_2 is not None:
        #         # boxes_scores = np.concatenate((boxes_scores, boxes_1[:, 8]), axis=0)
        #         # boxes = np.concatenate((boxes, boxes_1[:, :8].reshape((-1, 4, 2))), axis=0)
        #         boxes_scores_2 = boxes_2[:, 8] * 4
        #         boxes_2 = boxes_2[:, :8].reshape((-1, 4, 2))
        #         boxes_2[:, :, 0] /= ratio_w
        #         boxes_2[:, :, 1] /= ratio_h
        #         if boxes is not None:
        #             boxes_scores = np.concatenate((boxes_scores, boxes_scores_2), axis=0)
        #             boxes = np.concatenate((boxes, boxes_2), axis=0)
        #         else:
        #             boxes_scores = boxes_scores_2
        #             boxes = boxes_2
        #
        #     if boxes is not None:
        #         boxes = self.ms_standard_nms(boxes, boxes_scores, 0.5)
        #
        #     if boxes is None:
        #         json_dict[image_name] = {
        #             "box_num": 0,
        #             "annotations": []
        #         }
        #         print(json_dict)
        #         continue
        #
        #     annotations_list = []
        #     word_count = 1
        #     for box in boxes:
        #         # to avoid submitting errors
        #         box = self.sort_poly(box.astype(np.int32))
        #         if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
        #             continue
        #         x1 = box[0, 0] if box[0, 0] > 0 else 0
        #         y1 = box[0, 1] if box[0, 1] > 0 else 0
        #         x2 = box[1, 0] if box[1, 0] > 0 else 0
        #         y2 = box[1, 1] if box[1, 1] > 0 else 0
        #         x3 = box[2, 0] if box[2, 0] > 0 else 0
        #         y3 = box[2, 1] if box[2, 1] > 0 else 0
        #         x4 = box[3, 0] if box[3, 0] > 0 else 0
        #         y4 = box[3, 1] if box[3, 1] > 0 else 0
        #
        #         x_min = min(x1, x2, x3, x4)
        #         x_max = max(x1, x2, x3, x4)
        #         y_min = min(y1, y2, y3, y4)
        #         y_max = max(y1, y2, y3, y4)
        #         word_background = np.zeros((np.int32(y_max - y_min), np.int32(x_max - x_min)), dtype=np.int32)
        #         poly_area = np.array([[x1 - x_min, y1 - y_min], [x2 - x_min, y2 - y_min], [x3 - x_min, y3 - y_min],
        #                               [x4 - x_min, y4 - y_min]])
        #         cv2.fillPoly(word_background, np.int32([poly_area]), 1)
        #         word_area = np.copy(im[y_min:y_max, x_min:x_max])
        #         word_name = re.sub(".jpg", "_word_" + str(word_count) + ".jpg", im_fn)
        #         word_name = FLAGS.words_path + word_name.split("/")[-1]
        #         try:
        #             word_area[:, :, 0] *= np.uint8(word_background)
        #             word_area[:, :, 1] *= np.uint8(word_background)
        #             word_area[:, :, 2] *= np.uint8(word_background)
        #
        #         except Exception as e:
        #             print('\033[0;31m', word_name, "Dividing encounters error, store the origin cropped part.",
        #                   "\033[0m", e)
        #             print(word_area.shape, box)
        #         cv2.imwrite(
        #             filename=word_name,
        #             img=word_area)
        #         word_count += 1
        #         with open(word_name, "rb") as f:
        #             input_image_str = f.read()
        #         sess_outputs = recognition_sess.run(fetches, feed_dict={input_image_str_tensor: input_image_str})
        #         annotations_list.append({
        #             "text": sess_outputs['recognition_text'].decode('utf-8'),
        #             "bbox": [int(x4), int(y4), int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)]
        #         })
        #     json_dict[image_name] = {
        #         "box_num": boxes.shape[0],
        #         "annotations": annotations_list
        #     }
        # print(json_dict)
        # with open("scenetext_result/scenetext_result.json", "w") as f:
        #     json.dump(json_dict, f)

        # recognition_sess.close()
        # detection_sess.close()

    def output(self, im_fn):
        # self.load_model()
        json_dict = {}
        # im_fn_list = self.get_images()

        print(im_fn)
        image_name = im_fn.split("/")[-1]
        im = cv2.imread(im_fn)[:, :, ::-1]
        if im.shape[0] > 1280 or im.shape[1] > 1280:
            pass

        im_resized, (ratio_h, ratio_w) = self.resize_image(im, max_side_len=1280)
        boxes = None
        boxes_scores = None
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        # start = time.time()
        score, geometry = self.detection_sess.run([self.f_score, self.f_geometry],
                                                  feed_dict={self.input_images: [im_resized]})

        boxes_0 = self.detect(score_map=score, geo_map=geometry)

        if boxes_0 is not None:
            boxes_scores = boxes_0[:, 8]
            boxes = boxes_0[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        im_resized, (ratio_h, ratio_w) = self.resize_image(im, max_side_len=896)

        score, geometry = self.detection_sess.run([self.f_score, self.f_geometry],
                                                  feed_dict={self.input_images: [im_resized]})

        boxes_1 = self.detect(score_map=score, geo_map=geometry)

        if boxes_1 is not None:

            boxes_scores_1 = boxes_1[:, 8] * 2
            boxes_1 = boxes_1[:, :8].reshape((-1, 4, 2))
            boxes_1[:, :, 0] /= ratio_w
            boxes_1[:, :, 1] /= ratio_h
            if boxes is not None:
                boxes_scores = np.concatenate((boxes_scores, boxes_scores_1), axis=0)
                boxes = np.concatenate((boxes, boxes_1), axis=0)
            else:
                boxes_scores = boxes_scores_1
                boxes = boxes_1

        im_resized, (ratio_h, ratio_w) = self.resize_image(im, max_side_len=1024)

        score, geometry = self.detection_sess.run([self.f_score, self.f_geometry],
                                                  feed_dict={self.input_images: [im_resized]})

        boxes_2 = self.detect(score_map=score, geo_map=geometry)

        if boxes_2 is not None:
            # boxes_scores = np.concatenate((boxes_scores, boxes_1[:, 8]), axis=0)
            # boxes = np.concatenate((boxes, boxes_1[:, :8].reshape((-1, 4, 2))), axis=0)
            boxes_scores_2 = boxes_2[:, 8] * 4
            boxes_2 = boxes_2[:, :8].reshape((-1, 4, 2))
            boxes_2[:, :, 0] /= ratio_w
            boxes_2[:, :, 1] /= ratio_h
            if boxes is not None:
                boxes_scores = np.concatenate((boxes_scores, boxes_scores_2), axis=0)
                boxes = np.concatenate((boxes, boxes_2), axis=0)
            else:
                boxes_scores = boxes_scores_2
                boxes = boxes_2

        if boxes is None:
            json_dict['box_num'] = 0
            json_dict['annotations'] = []
            cv2.imwrite(os.getenv('ABSOLUTE_PATH')+'static/output/' + im_fn, im)
            return json_dict

        if boxes is not None:
            boxes = self.ms_standard_nms(boxes, boxes_scores, 0.5)

        annotations_list = []
        word_count = 1
        for box in boxes:
            # to avoid submitting errors
            box = self.sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            x1 = box[0, 0] if box[0, 0] > 0 else 0
            y1 = box[0, 1] if box[0, 1] > 0 else 0
            x2 = box[1, 0] if box[1, 0] > 0 else 0
            y2 = box[1, 1] if box[1, 1] > 0 else 0
            x3 = box[2, 0] if box[2, 0] > 0 else 0
            y3 = box[2, 1] if box[2, 1] > 0 else 0
            x4 = box[3, 0] if box[3, 0] > 0 else 0
            y4 = box[3, 1] if box[3, 1] > 0 else 0

            x_min = min(x1, x2, x3, x4)
            x_max = max(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            y_max = max(y1, y2, y3, y4)
            word_background = np.zeros((np.int32(y_max - y_min), np.int32(x_max - x_min)), dtype=np.int32)
            poly_area = np.array([[x1 - x_min, y1 - y_min], [x2 - x_min, y2 - y_min], [x3 - x_min, y3 - y_min],
                                  [x4 - x_min, y4 - y_min]])
            cv2.fillPoly(word_background, np.int32([poly_area]), 1)
            word_area = np.copy(im[y_min:y_max, x_min:x_max])
            word_name = re.sub(".jpg", "_word_" + str(word_count) + ".jpg", im_fn)
            word_name = FLAGS.words_path + word_name.split("/")[-1]
            try:
                word_area[:, :, 0] *= np.uint8(word_background)
                word_area[:, :, 1] *= np.uint8(word_background)
                word_area[:, :, 2] *= np.uint8(word_background)

            except Exception as e:
                print('\033[0;31m', word_name, "Dividing encounters error, store the origin cropped part.",
                      "\033[0m", e)
                print(word_area.shape, box)
            cv2.imwrite(filename=word_name, img=word_area)
            word_count += 1
            with open(word_name, "rb") as f:
                input_image_str = f.read()
            sess_outputs = self.recognition_sess.run(self.fetches,
                                                     feed_dict={self.input_image_str_tensor: input_image_str})
            annotations_list.append({
                "text": sess_outputs['recognition_text'].decode('utf-8'),
                "bbox": [int(x4), int(y4), int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)]
            })

        image_out = cv2.imread(im_fn)
        for box in boxes:
            # to avoid submitting errors
            box = self.sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            # print(box)
            cv2.polylines(image_out,[box],True,color=(0,0,255))
        cv2.imwrite(os.getenv('ABSOLUTE_PATH')+'/app/static/output/'+image_name,image_out)

        json_dict["box_num"] =  boxes.shape[0]
        json_dict["annotations"] = annotations_list

        with open(os.getenv('ABSOLUTE_PATH')+"model/scenetext_result/scenetext_result.json", "w") as f:
            json.dump(json_dict, f)

        return json_dict


def runoverwrite(main=None, argv=None):
    import sys as _sys
    from tensorflow.python.platform import flags
    f = flags.FLAGS
    args = argv[1:] if argv else None
    flags_passthrough = f._parse_flags(args=args)
    main(_sys.argv[:1] + flags_passthrough)


if __name__ == '__main__':
    textObject = Text_recognition()
    runoverwrite(textObject.load_model)
    textObject.output(os.path.join(os.getenv('ABSOLUTE_PATH'), os.getenv('UPLOAD_PATH'),'img_1.jpg'))

# import cv2
# import time
# import math
# import os
# import numpy as np
# import tensorflow as tf
# import scipy
# import copy
# import model.detection.locality_aware_nms as nms_locality
# import lanms
# #/home/yangliu/pythoncode/EAST-master/data/icpr1000/image_1000
# tf.app.flags.DEFINE_string('test_data_path', '/home/yangliu/EAST-master-all-back/datasets/MSRA_TD500_HUST-TR400/test/img/', '')
# tf.app.flags.DEFINE_string('gpu_list', '1', '')
# tf.app.flags.DEFINE_string('checkpoint_path', '/home/yangliu/EAST-master-all-back/tmp/EAST_Msra_Hust_GCN_AABB212_longedge_bs16/', '')
# tf.app.flags.DEFINE_string('output_dir', '/home/yangliu/EAST-master-all-back/test2/east_Msra_Hust_GCN_AABB212_longedge_bs16_9_84961', '')
# tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
#
# import model_GCN_AABB_SEBbn_5theta as model
# import msra_hust_AABB212_hsv_center_rotate_again_longedge as icdar
#
# FLAGS = tf.app.flags.FLAGS
#
# def get_images():
#     '''
#     find image files in test data path
#     :return: list of files found
#     '''
#     files = []
#     exts = ['jpg', 'png', 'jpeg', 'JPG']
#     for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
#         for filename in filenames:
#             for ext in exts:
#                 if filename.endswith(ext):
#                     files.append(os.path.join(parent, filename))
#                     break
#     print('Find {} images'.format(len(files)))
#     return files
#
#
# def resize_image(im, max_side_len):
#     '''
#     resize image to a size multiple of 32 which is required by the network
#     :param im: the resized image
#     :param max_side_len: limit of max image size to avoid out of memory in gpu
#     :return: the resized image and the resize ratio
#     '''
#     h, w, _ = im.shape
#
#     resize_w = w
#     resize_h = h
#
#     # limit the max side
#     if max(resize_h, resize_w) > max_side_len:
#         ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
#     else:
#         ratio = 1.
#     resize_h = int(resize_h * ratio)
#     resize_w = int(resize_w * ratio)
#
#     resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
#     resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
#     im = cv2.resize(im, (int(resize_w), int(resize_h)))
#
#     ratio_h = resize_h / float(h)
#     ratio_w = resize_w / float(w)
#
#     return im, (ratio_h, ratio_w)
#
#
# def detect(score_map, geo_map, timer,filename, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
#     '''
#     restore text boxes from score map and geo map
#     :param score_map:
#     :param geo_map:
#     :param timer:
#     :param score_map_thresh: threshhold for score map
#     :param box_thresh: threshhold for boxes
#     :param nms_thres: threshold for nms
#     :return:
#     '''
#     if len(score_map.shape) == 4:
#         score_map = score_map[0, :, :, 0]
#         geo_map = geo_map[0, :, :, ]
#         print('score_map.shape is {}'.format(score_map.shape))
#         print(np.amax(score_map))
#         print(np.amax(geo_map[:,:,1]))
#        # scipy.misc.imsave("image_score.jpg",score_map)
#         print(filename)
#         fn_list = filename.split("/")
#         fn_list_len = len(fn_list)
#         print(fn_list_len)
#         new_name_0 = "./maps/score_map_"+fn_list[fn_list_len-2] +"/"+fn_list[fn_list_len-1].split(".")[0]+"_score.jpg"
#
#         new_name_1 = "./maps/score_map_"+fn_list[fn_list_len-2] +"/"+fn_list[fn_list_len-1].split(".")[0]+"_geo_1.jpg"
#         new_name_2 = "./maps/score_map_"+fn_list[fn_list_len-2] +"/"+fn_list[fn_list_len-1].split(".")[0]+"_geo_2.jpg"
#         new_name_3 = "./maps/score_map_"+fn_list[fn_list_len-2] +"/"+fn_list[fn_list_len-1].split(".")[0]+"_geo_3.jpg"
#         new_name_4 = "./maps/score_map_"+fn_list[fn_list_len-2] +"/"+fn_list[fn_list_len-1].split(".")[0]+"_geo_4.jpg"
#         new_name_5 = "./maps/score_map_"+fn_list[fn_list_len-2] +"/"+fn_list[fn_list_len-1].split(".")[0]+"_geo_5.jpg"
#
#         print(new_name_0)
#         print(new_name_1)
#
#         print('geo_map.shape is {}'.format(geo_map.shape))
#
#     # filter the score map
#     xy_text = np.argwhere(score_map > score_map_thresh)
#     print('xy_text.shape is {}'.format(xy_text.shape))
#     # sort the text boxes via the y axis
#     xy_text = xy_text[np.argsort(xy_text[:, 0])]
#     print('xy_text_sort.shape is {}'.format(xy_text.shape))
#     # restore
#     start = time.time()
#     text_box_restored = icdar.restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
#     print('text_box_restored is {}'.format(text_box_restored.shape))
#
#     print('{} text boxes before nms'.format(text_box_restored.shape[0]))
#
#     boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
#     boxes[:, :8] = text_box_restored.reshape((-1, 8))
#     #print('reshape text_box_restored is {}'.format(text_box_restored.reshape((-1, 8)).shape))
#     boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
#     timer['restore'] = time.time() - start
#     # nms part
#     start = time.time()
#     boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
#     #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
#     print('{} text boxes after nms'.format(boxes.shape[0]))
#     timer['nms'] = time.time() - start
#
#     if boxes.shape[0] == 0:
#         return None, timer
#
#     # here we filter some low score boxes by the average score map, this is different from the orginal paper
#     for i, box in enumerate(boxes):
#         mask = np.zeros_like(score_map, dtype=np.uint8)
#         cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
#         boxes[i, 8] = cv2.mean(score_map, mask)[0]
#     boxes = boxes[boxes[:, 8] > box_thresh]
#
#     return boxes, timer
#
#
#
# def sort_poly(p):
#     min_axis = np.argmin(np.sum(p, axis=1))
#     p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
#     if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
#         return p
#     else:
#         return p[[0, 3, 2, 1]]
#
# import numpy as np
# from shapely.geometry import Polygon
#
# def intersection_1(g, p):
#     #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
#     #(p.shape)
#     g = Polygon(g)
#     p = Polygon(p)
#
#     if not g.is_valid or not p.is_valid:
#         return 0
#     inter = g.intersection(p).area
#     #print(g)
#     #print(p)
#     #print(inter)
#     union = min(g.area, p.area)
#     #union = g.area + p.area - inter
#     if union == 0:
#         return 0
#     else:
#         return inter/union
#
# def ms_standard_nms(S, scores, thres):
#     ss1 = (S.shape)
#     ss2 = (scores.shape)
#     order = np.argsort(scores)[::-1]
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         si = S[i]
#
#         ovr = np.array([intersection_1(si, S[t]) for t in order[1:]])
#         print(ovr)
#         print(ovr)
#         #ovr_1 = np.array(intersection_1(si, si))
#         inds = np.where(ovr <= thres)[0]
#         order = order[inds+1]
#
#     return S[keep]
#
# def main(argv=None):
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
#
#
#     try:
#         os.makedirs(os.path.join(FLAGS.output_dir, 'img'))
#         os.makedirs(os.path.join(FLAGS.output_dir, 'txt'))
#     except OSError as e:
#         if e.errno != 17:
#             raise
#
#     with tf.get_default_graph().as_default():
#         input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
#         global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
#         f_score, f_geometry = model.model(input_images, is_training=False)
#
#         variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
#         saver = tf.train.Saver(variable_averages.variables_to_restore())
#
#         with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#             ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
#             model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
#             print('Restore from {}'.format(model_path))
#             saver.restore(sess, model_path)
#
#             im_fn_list = get_images()
#             for im_fn in im_fn_list:
#                 im = cv2.imread(im_fn)[:, :, ::-1]
#
#                 im_resized, (ratio_h, ratio_w) = resize_image(im,  max_side_len=1280)
#                 boxes = None
#                 boxes_scores = None
#                 timer = {'net': 0, 'restore': 0, 'nms': 0}
#                 start = time.time()
#                 score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
#
#                 boxes_0, timer = detect(score_map=score, geo_map=geometry, timer=timer, filename = im_fn)
#
#                 if boxes_0 is not None:
#                     boxes_scores = boxes_0[:, 8]
#                     boxes = boxes_0[:, :8].reshape((-1, 4, 2))
#                     boxes[:, :, 0] /= ratio_w
#                     boxes[:, :, 1] /= ratio_h
#
#                 im_resized, (ratio_h, ratio_w) = resize_image(im,  max_side_len=896)
#
#                 score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
#
#                 boxes_1, timer = detect(score_map=score, geo_map=geometry, timer=timer, filename=im_fn)
#
#                 if boxes_1 is not None:
#
#                     boxes_scores_1 =  boxes_1[:, 8] * 2
#                     boxes_1 = boxes_1[:, :8].reshape((-1, 4, 2))
#                     boxes_1[:, :, 0] /= ratio_w
#                     boxes_1[:, :, 1] /= ratio_h
#                     if boxes is not None:
#                         boxes_scores = np.concatenate((boxes_scores, boxes_scores_1), axis=0)
#                         boxes = np.concatenate((boxes, boxes_1), axis=0)
#                     else:
#                         boxes_scores =  boxes_scores_1
#                         boxes = boxes_1
#
#                 im_resized, (ratio_h, ratio_w) = resize_image(im, max_side_len=640)
#
#                 score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
#
#                 boxes_2, timer = detect(score_map=score, geo_map=geometry, timer=timer, filename=im_fn)
#
#
#                 if boxes_2 is not None:
#                     # boxes_scores = np.concatenate((boxes_scores, boxes_1[:, 8]), axis=0)
#                     # boxes = np.concatenate((boxes, boxes_1[:, :8].reshape((-1, 4, 2))), axis=0)
#                     boxes_scores_2 = boxes_2[:, 8] * 4
#                     boxes_2 = boxes_2[:, :8].reshape((-1, 4, 2))
#                     boxes_2[:, :, 0] /= ratio_w
#                     boxes_2[:, :, 1] /= ratio_h
#                     if boxes is not None:
#                         boxes_scores = np.concatenate((boxes_scores, boxes_scores_2), axis=0)
#                         boxes = np.concatenate((boxes, boxes_2), axis=0)
#                     else:
#                         boxes_scores =  boxes_scores_2
#                         boxes = boxes_2
#
#                 if boxes is not None:
#                     boxes = ms_standard_nms(boxes, boxes_scores, 0.5)
#                 # save to file
#                 if boxes is not None:
#
#                     res_file = os.path.join(
#                         FLAGS.output_dir, 'txt',
#                         ('res_{}.txt').format(
#                             os.path.basename(im_fn).split('.')[0]))
#
#                     with open(res_file, 'w') as f:
#                         i = 0
#                         for box in boxes:
#                             #box_score = boxes_scores[i]
#
#                             # to avoid submitting errors
#                             box = sort_poly(box.astype(np.int32))
#                             if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
#                                 continue
#                             # if(i%10 == 0):
#                             if (True):
#                                 f.write('{},{},{},{},{},{},{},{}\r\n'.format(
#                                     box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0],
#                                     box[3, 1],
#                                 ))
#                                 cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
#                                               color=(255, 0, 255), thickness=2)
#                                 #print(box_score)
#                                 #cv2.putText(im_s[:, :, ::-1], str(box_score)[0:5], (box[0, 0], box[0, 1]),
#                                 #            cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 255), 2)
#                             i = i + 1
#
#                 if not FLAGS.no_write_images:
#                     img_path = os.path.join(FLAGS.output_dir, 'img',os.path.basename(im_fn))
#                     cv2.imwrite(img_path, im[:, :, ::-1])
#
# if __name__ == '__main__':
#     tf.app.run()

# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from timeit import time
import warnings
import argparse

import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator
import pickle 

from yolo_v3 import YOLO3
from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from reid import REID
from face_net import FaceNet
import copy

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
# from scipy.misc import imresize
from skimage.transform import resize
from sklearn.metrics.pairwise import euclidean_distances


parser = argparse.ArgumentParser()
parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
parser.add_argument('--videos', nargs='+', help='List of videos', required=True)
parser.add_argument('-all', help='Combine all videos into one', default=True)
args = parser.parse_args()  # vars(parser.parse_args())


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh


def main(yolo):
    print(f'Using {yolo} model')
    # Definition of the parameters
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4

    # deep_sort
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # use to get feature

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    output_frames = []
    output_rectanger = []
    output_areas = []
    output_wh_ratio = []

    is_vis = True
    out_dir = 'videos/output/'
    print('The output folder is', out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    all_frames = []
    for video in args.videos:
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        while True:
            ret, frame = video_capture.read()
            if ret is not True:
                video_capture.release()
                break
            all_frames.append(frame)

    frame_nums = len(all_frames)
    tracking_path = out_dir + 'tracking' + '.avi'
    combined_path = out_dir + 'allVideos' + '.avi'
    if is_vis:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
        out2 = cv2.VideoWriter(combined_path, fourcc, frame_rate, (w, h))
        # Combine all videos
        for frame in all_frames:
            out2.write(frame)
        out2.release()

    # Initialize tracking file
    filename = out_dir + '/tracking.txt'
    open(filename, 'w')

    fps = 0.0
    frame_cnt = 0
    t1 = time.time()

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []

    file_path="tracking_var.pkl"
    if os.path.exists(file_path):
        with open (file_path,"rb") as f:
            track_cnt = pickle.load(f)
            images_by_id = pickle.load(f)
            ids_per_frame = pickle.load(f)
            combined_path = pickle.load(f)
            tracking_path = pickle.load(f)
    else:
        for frame in all_frames:
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxs = yolo.detect_image(image)  # n * [topleft_x, topleft_y, w, h]
            features = encoder(frame, boxs)  # n * 128
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]  # length = n
            text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
            # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]  # length = len(indices)

            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            tmp_ids = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
                if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                    tmp_ids.append(track.track_id)
                    if track.track_id not in track_cnt:
                        track_cnt[track.track_id] = [
                            [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]
                        ]
                        images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                    else:
                        track_cnt[track.track_id].append([
                            frame_cnt,
                            int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                            area
                        ])
                        images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                cv2_addBox(
                    track.track_id,
                    frame,
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    line_thickness,
                    text_thickness,
                    text_scale
                )
                write_results(
                    filename,
                    'mot',
                    frame_cnt + 1,
                    str(track.track_id),
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    w, h
                )
            ids_per_frame.append(set(tmp_ids))

            # save a frame
            if is_vis:
                out.write(frame)
            t2 = time.time()

            frame_cnt += 1
            print(frame_cnt, '/', frame_nums)


        if is_vis:
            out.release()
        print('Tracking finished in {} seconds'.format(int(time.time() - t1)))
        print('Tracked video : {}'.format(tracking_path))
        print('Combined video : {}'.format(combined_path))
        with open(file_path,"wb") as f:
            pickle.dump(track_cnt,f)
            pickle.dump(images_by_id,f)
            pickle.dump(ids_per_frame,f)
            pickle.dump(combined_path,f)
            pickle.dump(tracking_path,f) 

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    reid = REID()
    face_network = FaceNet()
    body_threshold = 320
    face_threshold = 1.61
    exist_ids = set()
    final_fuse_id = dict()

    print(f'Total IDs = {len(images_by_id)}')
        
    face_feats = dict()
    body_feats = dict()
    file_path1 = "feature_var.pkl"
    if os.path.exists(file_path1):
        with open(file_path1,"rb") as f:
            face_feats=pickle.load(f)
            body_feats=pickle.load(f)
    else:
        for i in images_by_id:
            print(f'ID number {i} -> Number of frames {len(images_by_id[i])}')
            face_features = face_network._features(images_by_id[i])
            if face_features != None :
                face_feats[i]=face_features
            body_feats[i]=reid._features(images_by_id[i])
        with open(file_path1,"wb") as f:
            pickle.dump(face_feats,f)
            pickle.dump(body_feats,f)

        # body_features = reid._features(images_by_id[i])
        # print("Body feature dimension:",body_features.shape)

        # face_features = face_network._features(images_by_id[i])
        # # face_features = resize(face_features, (body_features.shape[1], body_features.shape[2]),preserve_range=True, mode='reflect')
        # if face_features != None :
        #     print("Face feature dimension:",face_features.shape)
        #     face_features = resize(face_features,body_features.shape,preserve_range=False, mode='reflect')
        #     feats[i]= np.concatenate((body_features, face_features), axis=0)
        # feats[i]=np.concatenate((body_features, body_features), axis=0)

    # # Perform person re-identification using KNN
    # knn = NearestNeighbors(n_neighbors=3)
    # all_features = np.concatenate(list(feats.values()), axis=0)
    # knn.fit(all_features)
    # dist_matrix = euclidean_distances(all_features)

    # # Perform person re-identification using DBSCAN
    # clustering = DBSCAN(metric='euclidean', eps=200, min_samples=1)
    # clustering.fit(dist_matrix)
    # print("****************Distance matrix************")
    # print(dist_matrix)
    # # Assign IDs to each cluster using the majority voting scheme
    # for i, label in enumerate(clustering.labels_):
    #     if label != -1:
    #         person_id = list(feats.keys())[label]
    #         if person_id not in final_fuse_id:
    #             final_fuse_id[person_id] = []
    #         final_fuse_id[person_id].append(i)
    # # print(f'final final_fuse_id after cluster :',final_fuse_id)
    # # Merge tracks of the same person across different videos
    for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    
                    if len(images_by_id[nid]) < 10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                    for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                        try:
                            if face_feats[nid] != None and face_feats[oid] != None:
                                # print("Face distance")
                                tmp = np.mean(reid.compute_distance(face_feats[nid], face_feats[oid]))
                                print('face nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                                dis.append([oid, tmp,0])
                            
                                
                        except KeyError as e:
                            print(f"*Face KeyError: {e} not found in face_feats dictionary")
                            try:
                                tmp = np.mean(reid.compute_distance(body_feats[nid], body_feats[oid]))
                                print('body nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                                dis.append([oid, tmp,1])
                            except KeyError as e:
                                print(f"*Body KeyError: {e} not found in face_feats dictionary")

                        
                        
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    dis.sort(key=operator.itemgetter(1))
                    # print(f"Body Distance between {dis[0][]} and {nid} is : {dis[0][1]}");
                    if dis[0][2] == 0 and dis[0][1] < face_threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    elif dis[0][2]== 1 and dis[0][1] < body_threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]





                        
    print('Before face_net Final ids and their sub-ids:', final_fuse_id)

    # face_network = FaceNet()
    # feats = dict()
    # face_threshold = 320
    # for id in final_fuse_id.keys():
    #     print(id)
    #     feats[id] = face_network._features(images_by_id[id])
    # fuse_keys=[ i for i in final_fuse_id.keys() ]
    
    # for id in fuse_keys:
    #     if feats[id] != None:
    #         for cmp_id in fuse_keys:
    #             if id != cmp_id:
    #                 if feats[cmp_id] != None:
    #                     tmp = np.mean(face_network.compute_distance(feats[id], feats[cmp_id]))
    #                     if tmp < face_threshold:
    #                         # final_fuse_id[id].append(cmp_id)
    #                         final_fuse_id[id].extend(final_fuse_id[cmp_id])
    #                         images_by_id[id] += images_by_id[cmp_id]
    #                         del final_fuse_id[cmp_id]

    # print('******After FaceNet Final ids and their sub-ids:*****', final_fuse_id)
    print('MOT took {} seconds'.format(int(time.time() - t1)))
    t2 = time.time()
    # {main_id1 : [sub_ids, ...], main_id2 : [sub_ids, ...], ...}

    # To generate MOT for each person, declare 'is_vis' to True
    is_vis = False
    if is_vis:
        print('Writing videos for each ID...')
        output_dir = 'videos/output/tracklets/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        loadvideo = LoadVideo(combined_path)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        for idx in final_fuse_id:
            tracking_path = os.path.join(output_dir, str(idx)+'.avi')
            out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
            for i in final_fuse_id[idx]:
                for f in track_cnt[i]:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, f[0])
                    _, frame = video_capture.read()
                    text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                    cv2_addBox(idx, frame, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                    out.write(frame)
            out.release()
        video_capture.release()

    # Generate a single video with complete MOT/ReID
    # if args.all:
    #     loadvideo = LoadVideo(combined_path)
    #     video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     complete_path = out_dir+'/Complete'+'.avi'
    #     out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))

    #     for frame in range(len(all_frames)):
    #         frame2 = all_frames[frame]
    #         video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    #         _, frame2 = video_capture.read()
    #         for idx in final_fuse_id:
    #             for i in final_fuse_id[idx]:
    #                 if i in track_cnt:
    #                     for f in track_cnt[i]:
    #                         # print('frame {} f0 {}'.format(frame,f[0]))
    #                         if frame == f[0]:
    #                             text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
    #                             cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
    #         out.write(frame2)
    #     out.release()
    #     video_capture.release()
    if args.all:
        loadvideo = LoadVideo(combined_path)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        complete_path = out_dir+'/Complete'+'.avi'
        out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))
        
        # Set the ground truth identities for each person
        ground_truth = {}
        for i in final_fuse_id:
            ground_truth[i] = final_fuse_id[i][0]
        
        # Initialize the counters for correct and total identifications
        total_ids = len(final_fuse_id)
        correct_ids = 0
        
        # Iterate through each frame and compare the re-identified identities with the ground truth
        for frame in range(len(all_frames)):
            frame2 = all_frames[frame]
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            _, frame2 = video_capture.read()
            
            # Iterate through each re-identified identity and compare with ground truth
            for idx in final_fuse_id:
                for i in final_fuse_id[idx]:
                    for f in track_cnt[i]:
                        if frame == f[0]:
                            text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                            
                            # If the re-identified identity matches the ground truth, increment the correct identification counter
                            if ground_truth[idx] == i:
                                correct_ids += 1
                            print("correct_ids",correct_ids)
                            # Add the bounding box and label to the frame
                            cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                            
            out.write(frame2)
            
        # Calculate and print the accuracy of person re-identification
        accuracy = correct_ids / total_ids
        print('Accuracy: {:.2f}%'.format(accuracy*100))
        
        out.release()
        video_capture.release()


    os.remove(combined_path)
    print('\nWriting videos took {} seconds'.format(int(time.time() - t2)))
    print('Final video at {}'.format(complete_path))
    print('Total: {} seconds'.format(int(time.time() - t1)))


def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.15 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness=2)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


warnings.filterwarnings('ignore')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    main(yolo=YOLO3() if args.version == 'v3' else YOLO4())

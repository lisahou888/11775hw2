#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import pandas as pd


def store_surf_feat(surf_feat, surf_feat_path):
    # store as a panda compressed csv
    if len(surf_feat) == 0:
        return
    df = pd.DataFrame.from_records(surf_feat)
    df.to_csv(surf_feat_path, compression = 'zip', index_label = False)
    

def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    
    surf_feat = []
    for keyframe in get_keyframes(downsampled_video_filename, keyframe_interval):
        key_points, feat = surf.detectAndCompute(keyframe, None)
        if feat is None:
            continue
        # pdb.set_trace()
        if len(feat.shape)==1:
            feat = np.expand_dims(feat, axis=0)

        print(feat.shape)
        surf_feat.append(feat)
    store_surf_feat(surf_feat, surf_feat_video_filename)


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    print('keyframe_interval=' + str(keyframe_interval))
    hessian_threshold = my_params.get('hessian_threshold')
    print('hessian_threshold=' + str(hessian_threshold))
    surf_features_folderpath = my_params.get('surf_features')
    print('surf_features_folderpath=' + surf_features_folderpath)
    downsampled_videos = my_params.get('downsampled_videos')
    print('downsampled_videos=' + downsampled_videos)

    # TODO: Create SURF object
    surf = cv2.SURF(hessian_threshold)

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes
    # for video in os.listdir(downsampled_videos):
    #     surf_feat_video_filename = video[0:video.find('.')] + '.surf'
    #     get_surf_features_from_video(video, surf_features_folderpath + '/' + surf_feat_video_filename)

    fread = open(all_video_names, "r")
    for line in fread.readlines():
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

        if not os.path.isfile(downsampled_video_filename):
            continue

        # Get SURF features for one video
        get_surf_features_from_video(downsampled_video_filename,
                                     surf_feat_video_filename, keyframe_interval)

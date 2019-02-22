import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
])

alexnet = models.alexnet(pretrained=True)
new_model = nn.Sequential(*list(alexnet.features.children())[:-9])

def store_cnn_feat(cnn_feat, cnn_feat_path, compress_mode):
    # store as a panda compressed csv
    print("Extraction finished, saving...")
    if cnn_feat is None:
        return
    df = pd.DataFrame.from_records(cnn_feat)
    df.to_csv(cnn_feat_path, compression = compress_mode, index_label = False)


def get_cnn_features_from_video(downsampled_video_filename, cnn_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    
    cnn_feat = None
    for keyframe in get_keyframes(downsampled_video_filename, keyframe_interval):
        new_img = Image.fromarray(keyframe)
        img_tensor = preprocess(new_img).unsqueeze_(0)
        img_variable = torch.autograd.Variable(img_tensor)
        out = new_model(img_variable)
        feat = (torch.sum(out, dim=1)/out.shape[1]).squeeze().detach().numpy()
        feat = np.expand_dims(feat.flatten(),axis=0)

        if cnn_feat is None:
            cnn_feat = feat
        else:
            cnn_feat = np.concatenate((cnn_feat, feat), axis=0)

    if cnn_feat is None:
        no_feat.append(downsampled_video_filename)
    return cnn_feat


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
    cnn_features_folderpath = my_params.get('cnn_path')
    print('cnn_features_folderpath=' + cnn_features_folderpath)
    downsampled_videos = my_params.get('downsampled_videos')
    print('downsampled_videos=' + downsampled_videos)
    compress_mode = my_params.get('compress_mode')
    print('compress_mode=' + compress_mode)


    # Check if folder for SURF features exists
    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    for line in fread.readlines():
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        cnn_feat_video_filename = os.path.join(cnn_features_folderpath, video_name + '.cnn')

        if not os.path.isfile(downsampled_video_filename):
            continue

        # Get SURF features for one video
        print("******************************")
        print(video_name)
        print("******************************")
        cnn_feat = get_cnn_features_from_video(downsampled_video_filename,
                                     cnn_feat_video_filename, keyframe_interval)
        store_cnn_feat(cnn_feat, cnn_feat_video_filename, compress_mode)

    print("These files don't have features")
    print(no_feat)
    print("Complete!")
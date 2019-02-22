#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import pickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: model_file feat_dir feat_dim output_file")
        print("model_file -- path of the trained svm file")
        print("feat_dir -- dir of feature files")
        print("feat_dim -- dim of features; provided just for debugging")
        print("output_file -- path to save the prediction score")
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    clf = pickle.load(open(model_file, 'rb'))
    fread = open("../all_test.video", "r")
    fw = open(output_file, "w")

    for line in fread.readlines():
        line = line.replace('\n', '')
        feat_path = feat_dir + line + "." + feat_dir.replace('/', '')
        if os.path.exists(feat_path) == False:
            fw.write("0\n")
            continue
        feat_lines = open(feat_path, 'r').readlines()
        features = np.asarray([np.array(s.split(";")).astype('float32') for s in feat_lines], dtype=np.float32)
        if np.sum(features) != 0:
            features = features / np.sum(features)

        # predicting output
        pred = clf.predict_proba(features)
        fw.write(str(pred[0][1]) + "\n")

    print('SVM tested successfully for event %s!' % (model_file))





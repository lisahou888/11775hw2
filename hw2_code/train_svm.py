#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import pickle
import sys


# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: {0} event feature_path dim train_only_model_output model_output".format(sys.argv[0]))
        exit(1)


    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = sys.argv[3]
    train_input_path = '../all_trn.lst'
    val_input_path = '../all_val.lst'
    train_only_model_out = argv[4]
    model_out = argv[5]

    fread = open(train_input_path, "r")

    matrix_exist = 0

    # read the files one big np matrix
    for line in fread.readlines():
        line_label = line.split(" ")
        label = line_label[1].replace('\n', '')
        if label != event_name:
            label = 0
        else:
            label = 1
        feat_path = feat_dir + line_label[0] + "." + feat_dir.replace('/', '')
        if os.path.exists(feat_path) == False:
            continue
        feat_lines = open(feat_path, 'r').readlines()
        features = np.asarray([np.array(s.split(";")).astype('float32') for s in feat_lines], dtype=np.float32)
        # normalize features
        if np.sum(features) != 0:
            features = features / np.sum(features)
        if matrix_exist == 0:
            X = features
            y = np.array(label)
            matrix_exist = 1
            continue
        else:
            X = np.vstack((X, features))
            y = np.vstack((y, np.array(label)))


    # start training, class_weight can be tuned
    clf = SVC(gamma='scale', probability = True, class_weight = {1:5})
    # if event_name == "P002" or event_name == "P003":
    #     clf = SVC(gamma='scale', probability = True, class_weight = {1:5})
    # else:
    #     clf = SVC(gamma='scale', probability = True, class_weight = {1:10})

    clf.fit(X, y)
    pickle.dump(clf, open(train_only_model_out, 'wb'))
    fread.close()

    print('SVM trained successfully for event %s with training set only! ' % (event_name))



    fread = open(val_input_path, "r")

    # read the files one big np matrix
    for line in fread.readlines():
        line_label = line.split(" ")
        label = line_label[1].replace('\n', '')
        if label != event_name:
            label = 0
        else:
            label = 1
        feat_path = feat_dir + line_label[0] + "." + feat_dir.replace('/', '')
        if os.path.exists(feat_path) == False:
            continue
        feat_lines = open(feat_path, 'r').readlines()
        features = np.asarray([np.array(s.split(";")).astype('float32') for s in feat_lines], dtype=np.float32)
        # normalize features
        if np.sum(features) != 0:
            features = features / np.sum(features)

        X = np.vstack((X, features))
        y = np.vstack((y, np.array(label)))


    # start training, class_weight can be tuned
    clf = SVC(gamma='scale', probability = True, class_weight = {1:5})
    # if event_name == "P002" or event_name == "P003":
    #     clf = SVC(gamma='scale', probability = True, class_weight = {1:5})
    # else:
    #     clf = SVC(gamma='scale', probability = True, class_weight = {1:10})
    clf.fit(X, y)
    pickle.dump(clf, open(model_out, 'wb'))
    fread.close()

    print('SVM trained successfully for event %s with both training and validation sets! ' % (event_name))







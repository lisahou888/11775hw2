import numpy as np
import os
from sklearn.svm.classes import SVC
import pickle
import sys

if __name__ == '__main__':
    folder = sys.argv[1]
    output = sys.argv[2]
    kmeans_path = sys.argv[3]

    clf1 = pickle.load(open(folder + "svm.P001.model", 'rb'))
    clf2 = pickle.load(open(folder + "svm.P002.model", 'rb'))
    clf3 = pickle.load(open(folder + "svm.P003.model", 'rb'))

    fread = open("../all_test.video", "r")
    fw = open(output, "w")
    fw.write("VideoID,Label\n")

    for line in fread.readlines():
        line = line.replace('\n', '')
        feat_path = kmeans_path + line + ".kmeans"
        if os.path.exists(feat_path) == False:
            fw.write(line + ',' + "0\n")
            continue
        feat_lines = open(feat_path, 'r').readlines()
        features = np.asarray([np.array(s.split(";")).astype('float32') for s in feat_lines], dtype=np.float32)
        if np.sum(features) != 0:
            features = features / np.sum(features)

        # predicting P001
        pred1 = clf1.predict_proba(features)
        pred2 = clf2.predict_proba(features)
        pred3 = clf3.predict_proba(features)

        label = ""

        if pred1[0][1] == max(max(pred1[0][1], pred2[0][1]), pred3[0][1]):
            label = "1"
        if pred2[0][1] == max(max(pred1[0][1], pred2[0][1]), pred3[0][1]):
            label = "2"
        if pred3[0][1] == max(max(pred1[0][1], pred2[0][1]), pred3[0][1]):
            label = "3"
        fw.write(line + ',' + label + '\n')

    fw.close()



        
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import sys
import pickle
import yaml

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: {0} config_file selected_feat_path folder kmeans_model".format(sys.argv[0]))
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))


    all_video_names_path = my_params.get('all_video_names')
    print('all_video_names_path='+all_video_names_path)
    selected_feat_path = sys.argv[2]
    print('selected_feat_path=' + selected_feat_path)
    cnn_path = my_params.get('surf_path')
    print('cnn_path=' + cnn_path)
    cluster_num = int(my_params.get('kmeans_cluster_num'))
    print('cluster_num=' + str(cluster_num))
    compress_mode = my_params.get('compress_mode')
    print('compress_mode = ' + compress_mode)
    kmeans_path = sys.argv[3]
    print('kmeans_path=' + kmeans_path)
    kmeans_model = sys.argv[4]
    print('kmeans_model=' + kmeans_model)

    # train kmeans centers
    print("Importing selected features...")
    selected_features = pd.read_csv(selected_feat_path, compression=compress_mode).values
    print("selected features dimensions: ")
    print(selected_features.shape)
    print("Import complete!")

    # fit kmeans model
    # print("Fitting kmeans sklearn...")
    print("reading kmeans model...")
    # kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(selected_features)
    kmeans = pickle.load(open(kmeans_file, 'rb'))
    # pickle.dump(kmeans, open(kmeans_model, 'wb'))
    # print("Fitting complete! KMeans Model saved at location: " + kmeans_model)

    fread = open(all_video_names_path,"r")

    for line in fread.readlines():
        cnn_feat_path = cnn_path + '/' + line.replace('\n', '') + ".surf"
        # exception handling -- cnn file might not exist for every video
        if os.path.exists(cnn_feat_path) == False:
            continue

        if not os.path.exists(kmeans_path):
            os.mkdir(kmeans_path)

        kmeans_feat_out_path = kmeans_path + '/' + line.replace('\n', '') + ".kmeans"

        cnn_features = pd.read_csv(cnn_feat_path, compression = compress_mode).values

        print("Printing the cnn feature's dimensions...\n\n")
        print(cnn_features.shape)

        # predicting
        labels = kmeans.predict(cnn_features)

        # creating kmeans features
        kmeans_features = np.zeros(cluster_num)
        for i in range(labels.shape[0]):
            kmeans_features[labels[i]] = kmeans_features[labels[i]] + 1

        # writing kmeans features
        print("Writing kmeans_features for " + line + " at " + kmeans_feat_out_path)
        print(kmeans_features)

        fwrite = open(kmeans_feat_out_path, 'w')
        line = str(kmeans_features[0])
        for m in range(1, kmeans_features.shape[0]):
            line += ';' + str(kmeans_features[m])
        fwrite.write(line + '\n')
        fwrite.close()

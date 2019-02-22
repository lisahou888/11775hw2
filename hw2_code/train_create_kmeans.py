import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import sys
import yaml

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {0} config_file".format(sys.argv[0]))
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))


    all_video_names_path = my_params.get('all_video_names')
    print('all_video_names_path='+all_video_names_path)
    surf_selected_feat_path = my_params.get('kmeans_selected_surf_path')
    print('surf_selected_feat_path=' + surf_selected_feat_path)
    surf_path = my_params.get('surf_path')
    print('surf_path=' + surf_path)
    cluster_num = int(my_params.get('kmeans_cluster_num'))
    print('cluster_num=' + str(cluster_num))
    compress_mode = my_params.get('compress_mode')
    print('compress_mode = ' + compress_mode)
    kmeans_path = my_params.get('kmeans_path')
    print('kmeans_path=' + kmeans_path)
    kmeans_model = my_params.get('kmeans_model')
    print('kmeans_model=' + kmeans_model)

    # train kmeans centers
    print("Importing selected features...")
    selected_features = pd.read_csv(surf_selected_feat_path, compress=compress_mode).values
    print("selected features dimensions: ")
    print(selected_features.shape)
    print("Import complete!")

    # fit kmeans model
    print("Fitting kmeans sklearn...")
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(selected_features)
    pickle.dump(kmeans open(kmeans_model, 'wb'))
    print("Fitting complete! KMeans Model saved at location: " + kmeans_model)

    fread = open(all_video_names_path,"r")

    for line in fread.readlines():
        surf_feat_path = surf_path + '/' + line.replace('\n', '') + ".surf"
        # exception handling -- surf file might not exist for every video
        if os.path.exists(surf_feat_path) == False:
            continue

        if not os.path.exists(kmeans_path):
            os.mkdir(kmeans_path)

        kmeans_feat_out_path = kmeans_path + '/' + line.replace('\n', '') + ".kmeans"

        surf_features = pd.read_csv(surf_feat_path, compress = compress_mode).values

        print("Printing the surf feature's dimensions...\n\n")
        print(surf_features.shape)

        # predicting
        labels = kmeans.predict(surf_features)

        # creating kmeans features
        kmeans_features = np.zeros(cluster_num)
        for i in range(labels.shape[0]):
            kmeans_features[labels[i]] = kmeans_features[labels[i]] + 1

        # writing kmeans features
        print("Writing kmeans_features for")
        print(line)
        print(kmeans_features)

        fwrite = open(surf_feat_path, 'w')
        line = str(kmeans_features[0])
        for m in range(1, kmeans_features.shape[0]):
            line += ';' + str(kmeans_features[m])
        fwrite.write(line + '\n')
        fwrite.close()

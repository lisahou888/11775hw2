import numpy as np
import os
import yaml
import pandas as pd
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {0} config_file".format(sys.argv[0]))
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))


    all_video_names_path = my_params.get('all_video_names')
    print('all_video_names_path=' + all_video_names_path)
    cnn_path = my_params.get('cnn_path')
    print('cnn_path=' + cnn_path)
    output_file = my_params.get('kmeans_collected_cnn_path')
    print('output_file=' + output_file)
    compress_mode = my_params.get('compress_mode')
    print('compress_mode=' + compress_mode)

    fread = open(all_video_names_path,"r")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows

    total_array = None

    for line in fread.readlines():
        cnn_path = cnn_path + '/' + line.replace('\n', '') + ".cnn"
        # exception handling -- cnn file might not exist for every video
        if os.path.exists(cnn_path) == False:
            continue
        array = pd.read_csv(cnn_path, compress = compress_mode).values
        if total_array is None:
            total_array = array
        else:
            total_array = np.concatenate(total_array, array)
    
    fread.close()

    # dump the selected features
    df = pd.DataFrame.from_records(total_array)
    df.to_csv(output_file, compression = compress_mode, index_label = False)
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


    # all_video_names_path = my_params.get('all_video_names')
    all_video_names_path = './list/small.video'
    print('all_video_names_path=' + all_video_names_path)
    surf_path = my_params.get('surf_path')
    print('surf_path=' + surf_path)
    ratio = float(my_params.get('kmeans_select_ratio'))
    print('ratio=' + str(ratio))
    output_file = my_params.get('kmeans_selected_surf_path')
    print('output_file=' + output_file)
    compress_mode = my_params.get('compress_mode')
    print('compress_mode=' + compress_mode)

    fread = open(all_video_names_path,"r")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    np.random.seed(18877)

    total_array = None

    for line in fread.readlines():
        surf_file = surf_path + '/' + line.replace('\n', '') + ".surf"
        print("*****************")
        print(line)
        print("*****************")
        # exception handling -- surf file might not exist for every video
        if os.path.exists(surf_file) == False:
            print(surf_file + "not found")
            continue
        array = pd.read_csv(surf_file, compression = compress_mode).values
        np.random.shuffle(array)
        select_size = int(array.shape[0] * ratio)
        feat_dim = array.shape[1]

        array = array[:select_size]
        print("selected size: " + str(array.shape))

        if total_array is None:
            total_array = array
        else:
            total_array = np.vstack((total_array, array))

        print("total array size: " + str(total_array.shape))
    
    fread.close()

    # dump the selected features
    df = pd.DataFrame.from_records(total_array)
    df.to_csv(output_file, compression = compress_mode, index_label = False)

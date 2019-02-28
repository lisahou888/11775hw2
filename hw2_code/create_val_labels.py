fread = open('../all_val.list', "r")
fw1 = open('list/P001_val_label')
fw2 = open('list/P002_val_label')
fw3 = open('list/P003_val_label')
fw_dict = {'P001':fw1, 'P002':fw2, 'P003':fw3}

for line in fread.readlines():
    label = line.replace('\n', '').split(' ')[1]
    for key in fw_dict.keys():
        fw = fw_dict[key]
        if key == label:
            fw.write('1\n')
        else:
            fw.write('0\n')
fw1.close()
fw2.close()
fw3.close()

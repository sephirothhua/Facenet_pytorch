import os
import numpy as np


def data2txt(DataPath,SavePath,split = 0.9):
    dirs = os.listdir(DataPath)
    data = []
    for dir in dirs:
        file_dirs = os.listdir(os.path.join(DataPath, dir))
        for file_dir in file_dirs:
            pics = os.listdir(os.path.join(DataPath, dir, file_dir))
            pics_dir = [os.path.join(DataPath, dir, file_dir, x) for x in pics]
            data += pics_dir
    labels = [int(x.split('/')[2]) for x in data]
    np.random.seed(50)
    np.random.shuffle(data)
    np.random.seed(50)
    np.random.shuffle(labels)
    train_data = data[:int(len(data)*split)]
    train_label = labels[:int(len(data)*split)]
    val_data = data[int(len(data)*split):]
    val_label = labels[int(len(data)*split):]
    write_txt(SavePath,'train_data.txt',train_data)
    write_txt(SavePath, 'train_label.txt', train_label)
    write_txt(SavePath, 'val_data.txt', val_data)
    write_txt(SavePath, 'val_label.txt', val_label)


def write_txt(SavePath,Filename,SaveList):
    file = open(os.path.join(SavePath, Filename),'w')
    for id in SaveList:
        file.writelines(str(id)+'\n')
    file.close()

if __name__ == '__main__':
    data2txt('./head_data','./data_txt')
from torch.utils.data import Dataset
import os
from torchvision import transforms as T
from PIL import Image
from skimage import io
import numpy as np
import torch
class FaceDataset(Dataset):
    def __init__(self,DataPath,DataType="train",img_size=30,transform = None):
        assert DataType in ("train","val"),"Data type must be train or val"
        self.image_size = (img_size,img_size)
        if(DataType == "train"):
            data = open(os.path.join(DataPath,'train_data.txt')).readlines()
            data = [x.strip() for x in data]
            label = open(os.path.join(DataPath,'train_label.txt')).readlines()
            label = [int(x) for x in label]
        elif(DataType == "val"):
            data = open(os.path.join(DataPath,'val_data.txt')).readlines()
            data = [x.strip() for x in data]
            label = open(os.path.join(DataPath,'val_label.txt')).readlines()
            label = [int(x) for x in label]

        self.classes = max(label)
        data_dict = {}
        for i in range(1,self.classes+1):
            data_list = [x for j,x in enumerate(data) if label[j]==i]
            data_dict[i-1] = data_list

        self.data = self.generate_triplets(data_dict,self.classes,len(data))

        if transform is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if(DataType == "train"):
                self.transform = T.Compose([
                    T.Resize(img_size),
                    T.RandomHorizontalFlip(),
                    T.CenterCrop(img_size),
                    T.ToTensor(),
                    # normalize
                ])
            elif(DataType == "val"):
                self.transform = T.Compose([
                    T.Resize(img_size),
                    T.CenterCrop(img_size),
                    T.ToTensor(),
                    # normalize
                ])
        else:
            self.transform = transform

    @staticmethod
    def generate_triplets(data,classes,triplets_num):
        classes = list(range(classes))
        triplets = []
        for _ in range(triplets_num):
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(data[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)
            pos_list = data[pos_class]
            neg_list = data[neg_class]

            if len(data[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(data[pos_class]))
                ipos = np.random.randint(0, len(data[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(data[pos_class]))
            ineg = np.random.randint(0, len(data[neg_class]))
            triplets.append([pos_list[ianc], pos_list[ipos], neg_list[ineg],
                             pos_class, neg_class])
        return triplets


    def __getitem__(self, index):
        anc_path, pos_path, neg_path, pos_class, neg_class = self.data[index]

        anc_img   = Image.open(anc_path)
        pos_img   = Image.open(pos_path)
        neg_img   = Image.open(neg_path)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class, 'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])
        return sample

    def __len__(self):
        return len(self.data)


def get_dataloader(root_dir,batch_size, num_workers,image_size):
    face_dataset = {
        'train': FaceDataset(DataPath=root_dir,
                             DataType="train",
                             img_size=image_size),
        'valid': FaceDataset(DataPath=root_dir,
                             DataType="val",
                             img_size=image_size)}

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x in ['train', 'valid']}

    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size

class test_data():
    def __init__(self,DataPath,img_size=30,transform = None):
        self.image_size = (img_size,img_size)
        data = open(os.path.join(DataPath,'val_data.txt')).readlines()
        self.data = [x.strip() for x in data]
        label = open(os.path.join(DataPath,'val_label.txt')).readlines()
        self.label = [int(x) for x in label]
        self.transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            # normalize
        ])

    def get_data(self):
        data_index = np.random.choice(np.arange(len(self.data)),size=100)
        pic_array = []
        label_array = []
        for index in data_index:
            pic = Image.open(self.data[index])
            pic = self.transform(pic)
            pic_array.append(pic)
            label_array.append(self.label[index])
        # pic_array = torch.
        return pic_array,label_array





if __name__ == '__main__':
    # data = FaceDataset('./data_txt')
    # data.__getitem__(0)
    from tensorboardX import SummaryWriter
    data_path = './data_txt'
    log_dir = "./logs/face_try"
    writer = SummaryWriter(log_dir=log_dir)
    Data = test_data('./data_txt')
    pics,labels = Data.get_data()
    for i,pic in enumerate(pics):
        writer.add_image("Person {}/{}".format(labels[i],i),pic,0)
    for i,pic in enumerate(pics):
        writer.add_image("Person {}/{}".format(labels[i],i),pic,1)




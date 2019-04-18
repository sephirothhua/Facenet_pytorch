import torch
from Config import Config
from ONNX_model import FaceNetModel
# from face_model import FaceNetModel
from torchvision import transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
cfg = Config()
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = FaceNetModel(embedding_size = cfg.embedding_size).to(device)
model = FaceNetModel(embedding_size = cfg.embedding_size).to(device)
# model_train = torch.load('./test_model/checkpoint_epoch220.pth', map_location='cuda:0')
# model.load_state_dict(model_train['state_dict'])
model_dict = model.state_dict()
checkpoint = torch.load('./test_model/checkpoint_epoch220.pth',map_location='cuda:0')
checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
model_dict.update(checkpoint['state_dict'])
model.load_state_dict(model_dict)
model.eval()

# transform = T.Compose([
#     T.Resize(cfg.image_size),
#     T.CenterCrop(cfg.image_size),
#     T.ToTensor(),
#     # normalize
# ])

def detect(img):
    img = (img / 255.0).astype(np.float32)
    img = torch.from_numpy(img.transpose((2,0,1)))
    pred = model.forward(img.unsqueeze(0).to(device)).to(device)
    return pred.cpu().detach().numpy()

def cosine_distance(a, b, data_is_normalized=False):

    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)
    # a = np.array(a)
    # b = np.array(b)
    # dist = np.sqrt(np.sum(np.square(a - b),axis=1))
    # return dist
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def nn_cosine_distance(x, y):

    distances = cosine_distance(x, y)
    return distances.min(axis=0)*10

# result_man1 = detect(Image.open('/data/Project/Detect/feature_test/data/001/3/0_4.jpg')).cpu().detach().numpy()
# result_man2 = detect(Image.open('/data/Project/Detect/feature_test/data/001/3/0_4.jpg')).cpu().detach().numpy()
# result_woman1 = detect(Image.open('/data/Project/Detect/feature_test/data/002/1/0_8.jpg')).cpu().detach().numpy()

def get_feature(image): # bbox shape must be []
    features = detect(image)
    return features

def person_read(dir):
    file_list = {}
    for root,dirs,files in os.walk(dir):
        for dir in dirs:
            files = os.listdir(os.path.join(root,dir))
            files = [os.path.join(root,dir,file) for file in files]
            file_list[dir] = files
    return file_list

features = []
man_dict = person_read('/data/Project/Detect/feature_test/data/001/')
woman_dict = person_read('/data/Project/Detect/feature_test/data/002/')

for file in man_dict['3']:
    img = Image.open(file)
    img = np.array(img)
    img = cv2.resize(img,(cfg.image_size,cfg.image_size))
    feature = get_feature(img)
    features.append(np.squeeze(feature))

man_result = []
for file in man_dict['15']:
    img = Image.open(file)
    img = np.array(img)
    img = cv2.resize(img, (cfg.image_size, cfg.image_size))
    feature = get_feature(img)
    # features.append(np.squeeze(feature))

    man_result.append(np.squeeze(nn_cosine_distance(features,[np.squeeze(feature)])))
woman_result = []
for file in woman_dict['20']:
    img = Image.open(file)
    img = np.array(img)
    img = cv2.resize(img, (cfg.image_size, cfg.image_size))
    feature = get_feature(img)
    # features.append(np.squeeze(feature))

    woman_result.append(np.squeeze(nn_cosine_distance(features,[np.squeeze(feature)])))

plt.plot(range(0,len(man_result)),man_result,'b-')
plt.plot(range(len(man_result),len(man_result)+len(woman_result)),woman_result,'r-')
plt.show()




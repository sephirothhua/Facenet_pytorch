import google.protobuf
from PIL import Image
import numpy as np
import time
from onnx_tensorrt.tensorrt_engine import Engine
import tensorrt as trt
import torchvision.transforms as transforms
from Config import Config
import os
import cv2
import torch
import matplotlib.pyplot as plt


def pil_loader(path):
	with open(path,'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def process_image(img_path, H=56, W=56):
	# image = pil_loader(img_path) ##(H,W,C)
	# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	# 								 std=[0.229, 0.224, 0.225])
	test_transforms = transforms.Compose([
		# transforms.Resize(size=(int(H), int(W)), interpolation=3),
		transforms.ToTensor(),
	])

	image = Image.open(img_path)
	image = np.array(image)
	image = cv2.resize(image, (H, W))
	# image = (image / 255.0).astype(np.float32)
	image = Image.fromarray(image)
	image = test_transforms(image)
	image = image.unsqueeze(0) ##(1,H,W,C)
	return image


cfg = Config()

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
trt_engine = trt.utils.load_engine(G_LOGGER, 'engine/test_engine.engine')
CROWD_ENGINE = Engine(trt_engine)

# img_path = 'head_data/001/1/0_3.jpg'
#
# ims = process_image(img_path)
# np_ims = np.asarray(ims.data.cpu())
#
# start = time.time()
# result = CROWD_ENGINE.run([np_ims])
# print(result)
# print(time.time()-start)

def get_feature(img_path): # bbox shape must be []
	ims = process_image(img_path)
	np_ims = np.asarray(ims.data.cpu())
	features = CROWD_ENGINE.run([np_ims])
	return features[0]

def person_read(dir):
	file_list = {}
	for root,dirs,files in os.walk(dir):
		for dir in dirs:
			files = os.listdir(os.path.join(root,dir))
			files = [os.path.join(root,dir,file) for file in files]
			file_list[dir] = files
	return file_list
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
features = []
man_dict = person_read('/data/Project/Detect/feature_test/data/001/')
woman_dict = person_read('/data/Project/Detect/feature_test/data/002/')

for file in man_dict['3']:
	# img = Image.open(file)
	# img = np.array(img)
	# img = cv2.resize(img,(cfg.image_size,cfg.image_size))
	feature = get_feature(file)
	features.append(np.squeeze(feature))

man_result = []
for file in man_dict['15']:
	# img = Image.open(file)
	# img = np.array(img)
	# img = cv2.resize(img, (cfg.image_size, cfg.image_size))
	feature = get_feature(file)
	# features.append(np.squeeze(feature))

	man_result.append(np.squeeze(nn_cosine_distance(features,[np.squeeze(feature)])))
woman_result = []
for file in woman_dict['20']:
	# img = Image.open(file)
	# img = np.array(img)
	# img = cv2.resize(img, (cfg.image_size, cfg.image_size))
	feature = get_feature(file)
	# features.append(np.squeeze(feature))

	woman_result.append(np.squeeze(nn_cosine_distance(features,[np.squeeze(feature)])))

plt.plot(range(0,len(man_result)),man_result,'b-')
plt.plot(range(len(man_result),len(man_result)+len(woman_result)),woman_result,'r-')
plt.show()


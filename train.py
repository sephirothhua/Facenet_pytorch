# -*- coding: utf-8 -*-
# @Time    : 19-4-18 下午4:35
# @Author  : Altair
# @FileName: w.py
# @Software: PyCharm
# @email   : 641084377@qq.com
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.modules.distance import PairwiseDistance
import torchvision
from torchvision import transforms
from eval_metrics import evaluate, plot_roc
from face_model import FaceNetModel,TripletLoss
from dataset import get_dataloader,test_data
from Config import Config
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from lr_scheduler import WarmAndReduce_LR


######################### Set the configration #############################
# The device for train : change the cuda device for your train
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)
# The Configration for the model. Change the Config.py for your own set.
cfg = Config()
# The data path of your data.
data_path = './data_txt'
# The logger path for your log dir.
log_dir = "./logs/face_log001"
writer = SummaryWriter(log_dir=log_dir)
# The test data path of your data. The classifier results will be write in the logger.
TestData = test_data('./data_txt',img_size=cfg.image_size)
############################################################################

########################## The Loss Functions ##############################
TL_loss = TripletLoss(cfg.margin) # The triplet loss of model
CE_loss = nn.CrossEntropyLoss()   # The cross entropy loss of model
############################################################################


def train_valid(model, optimizer, scheduler, epoch, dataloaders, data_size):
    for phase in ['train', 'valid']:
    # One step for train or valid
        labels, distances = [], []
        triplet_loss_sum = 0.0
        crossentropy_loss_sum = 0.0
        accuracy_sum = 0.0
        triplet_loss_sigma = 0.0
        crossentropy_loss_sigma = 0.0
        accuracy_sigma = 0.0

        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):
            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)
            if(anc_img.shape[0]!=cfg.batch_size or
            pos_img.shape[0]!=cfg.batch_size or
            neg_img.shape[0]!=cfg.batch_size):
                print("Batch Size Not Equal")
                continue

            pos_cls = batch_sample['pos_class'].to(device)
            neg_cls = batch_sample['neg_class'].to(device)

            with torch.set_grad_enabled(phase == 'train'):
                try:
                    # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                    anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                    # choose the hard negatives only for "training"
                    pos_dist = l2_dist.forward(anc_embed, pos_embed)
                    neg_dist = l2_dist.forward(anc_embed, neg_embed)

                    all = (neg_dist - pos_dist < cfg.margin).cpu().numpy().flatten()
                    if phase == 'train':
                        hard_triplets = np.where(all == 1)
                        if len(hard_triplets[0]) == 0:
                            continue
                    else:
                        hard_triplets = np.where(all >= 0)
                        if len(hard_triplets[0]) == 0:
                            continue

                    anc_hard_embed = anc_embed[hard_triplets].to(device)
                    pos_hard_embed = pos_embed[hard_triplets].to(device)
                    neg_hard_embed = neg_embed[hard_triplets].to(device)

                    anc_hard_img = anc_img[hard_triplets].to(device)
                    pos_hard_img = pos_img[hard_triplets].to(device)
                    neg_hard_img = neg_img[hard_triplets].to(device)

                    pos_hard_cls = pos_cls[hard_triplets].to(device)
                    neg_hard_cls = neg_cls[hard_triplets].to(device)

                    anc_img_pred = model.forward_classifier(anc_hard_img).to(device)
                    pos_img_pred = model.forward_classifier(pos_hard_img).to(device)
                    neg_img_pred = model.forward_classifier(neg_hard_img).to(device)

                    triplet_loss = TL_loss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)
                    triplet_loss *= cfg.triplet_lambuda
                    predicted_labels = torch.cat([anc_img_pred, pos_img_pred, neg_img_pred])
                    true_labels = torch.cat([pos_hard_cls, pos_hard_cls,neg_hard_cls]).squeeze()
                    crossentropy_loss = CE_loss(predicted_labels,true_labels).to(device)
                    loss = triplet_loss + crossentropy_loss

                    if phase == 'train':
                        optimizer.zero_grad()
                        # triplet_loss.backward()
                        loss.backward()
                        optimizer.step()
                    if phase == 'valid':
                        pic_array,_ = TestData.get_data()
                        for i,pic in enumerate(pic_array):
                            pred = model.forward_classifier(pic.unsqueeze(0).to(device)).to(device)
                            pred = torch.argmax(pred, 1).cpu().numpy()
                            # print(pred)
                            writer.add_image("Person {}/{}".format(pred[0],i),pic,epoch)

                    _, predicted = torch.max(predicted_labels, 1)
                    correct = (predicted == true_labels).cpu().squeeze().sum().numpy()/(len(hard_triplets[0])*3)

                    dists = l2_dist.forward(anc_embed, pos_embed)
                    distances.append(dists.data.cpu().numpy())
                    labels.append(np.ones(dists.size(0)))

                    dists = l2_dist.forward(anc_embed, neg_embed)
                    distances.append(dists.data.cpu().numpy())
                    labels.append(np.zeros(dists.size(0)))

                    triplet_loss_sum += triplet_loss.item()
                    crossentropy_loss_sum += crossentropy_loss.item()
                    accuracy_sum += correct

                    triplet_loss_sigma      += triplet_loss.item()
                    crossentropy_loss_sigma += crossentropy_loss.item()
                    accuracy_sigma          += correct
                    if batch_idx % 10 == 0 and batch_idx!=0:
                        print('{} Inter {:4d}/{:4d} - Triplet Loss = {:.5f} - CrossEntropy Loss = {:.5f} - All Loss = {:.5f} - Accuaracy = {:.5f} len:{}'
                              .format(phase,batch_idx,len(dataloaders[phase]),
                                      triplet_loss_sigma/10,crossentropy_loss_sigma/10,
                                      (triplet_loss_sigma+crossentropy_loss_sigma)/10,
                                      accuracy_sigma/10,len(hard_triplets[0])))
                        triplet_loss_sigma        = 0
                        crossentropy_loss_sigma   = 0
                        accuracy_sigma            = 0
                except Exception as e:
                    print(e)
                    pass
        avg_triplet_loss = triplet_loss_sum / int(data_size[phase]/cfg.batch_size)
        avg_crossentropy_loss = crossentropy_loss_sum / int(data_size[phase]/cfg.batch_size)
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - CrossEntropy Loss  = {:.8f}'.format(phase, avg_crossentropy_loss))
        print('  {} set - All Loss           = {:.8f}'.format(phase, avg_triplet_loss+avg_crossentropy_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

        # 记录训练loss
        writer.add_scalars('Loss/Triplet Loss Group'.format(phase), {'{} triplet loss'.format(phase): avg_triplet_loss}, epoch)
        writer.add_scalars('Loss/Crossentropy Loss Group'.format(phase), {'{} crossentropy loss'.format(phase): avg_crossentropy_loss}, epoch)
        writer.add_scalars('Loss/All Loss Group'.format(phase), {'{} loss'.format(phase): avg_triplet_loss + avg_crossentropy_loss}, epoch)
        writer.add_scalars('Accuracy_group'.format(phase), {'{} accuracy'.format(phase): np.mean(accuracy)}, epoch)
        # 记录learning rate
        writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)

        with open('./log/{}_log_epoch{}.txt'.format(phase, epoch), 'w') as f:
            f.write(str(epoch) + '\t' +
                    str(np.mean(accuracy)) + '\t' +
                    str(avg_triplet_loss)+ '\t' +
                    str(avg_crossentropy_loss)+ '\t' +
                    str(avg_triplet_loss+avg_crossentropy_loss))

        if phase == 'train':
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       './log/checkpoint_epoch{}.pth'.format(epoch))
        else:
            plot_roc(fpr, tpr, figure_name='./log/roc_valid_epoch_{}.png'.format(epoch))



def main():
    model     = FaceNetModel(embedding_size = cfg.embedding_size, num_classes = cfg.num_classes).to(device)
    if cfg.use_warmup:
        optimizer = optim.Adam(model.parameters(), lr = cfg.start_learning_rate)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)
        scheduler = WarmAndReduce_LR(optimizer,cfg.base_learning_rate,cfg.num_epochs,
                                  use_warmup=cfg.use_warmup,
                                  start_learning_rate=cfg.start_learning_rate,
                                  warmup_epoch=cfg.warmup_epoch)
    else:
        optimizer = optim.Adam(model.parameters(), lr = cfg.base_learning_rate)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)
        scheduler = WarmAndReduce_LR(optimizer,cfg.base_learning_rate,cfg.num_epochs,
                                  use_warmup=cfg.use_warmup)
    if cfg.start_epoch != 0:
        checkpoint = torch.load('./log/checkpoint_epoch{}.pth'.format(cfg.start_epoch-1),map_location='cuda:0')
        print("Load weights from {}".format('./log/checkpoint_epoch{}.pth'.format(cfg.start_epoch-1)))
        if cfg.del_classifier:
            model_dict = model.state_dict()
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(checkpoint['state_dict'])
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])
    for epoch in range(cfg.start_epoch, cfg.num_epochs + cfg.start_epoch):
        # scheduler.step()
        print(80 * '=')
        print('Epoch [{}/{}] Learning Rate:{:8f}'.format(epoch, cfg.num_epochs + cfg.start_epoch - 1,scheduler.get_lr()[0]))

        data_loaders, data_size = get_dataloader(data_path,cfg.batch_size,cfg.num_workers,cfg.image_size)

        train_valid(model, optimizer, scheduler, epoch, data_loaders, data_size)

        print(80 * '=')

if __name__ == '__main__':
    main()

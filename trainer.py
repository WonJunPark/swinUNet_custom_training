import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from torchvision import transforms
from glob import glob
from utils2.dataset import BasicDataset
from torch.utils.data import random_split
from losses import dice_coeff
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # print("The length of train set is: {}".format(len(db_train)))
    #
    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)

    img_path_list = sorted(glob('../sample_preprocessing/data/*/*.png'))
    mask_path_list = sorted(glob('../sample_preprocessing/label/*/*.png'))

    dataset = BasicDataset(img_path_list, mask_path_list)


    val_percent = 20 / 100
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)


    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
    #                          worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    old_loss = 9999

    for epoch_num in tqdm(iterator):
        total_loss = 0
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            # # resize
            # a = image_batch[0]
            # a = np.transpose(a, (1,2,0))
            # plt.imshow(a)
            # plt.show()

            image_batch, label_batch = image_batch.cuda().float(), label_batch.cuda().float()

            outputs = model(image_batch)
            outputs = F.upsample(outputs, [512, 512], mode='bilinear')

            # loss_ce = ce_loss(outputs[0], label_batch[0].long())
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)

            outputs = F.softmax(outputs, dim=1)

            if iter_num % 100 == 0:
                logging.info('\nivh : %f / ich : %f' % (outputs[:,1,:,:].max(),outputs[:,2,:,:].max()))


            dice_loss1 = dice_coeff(outputs[:, 1, :, :], label_batch[:, :, :, :])
            dice_loss2 = dice_coeff(outputs[:, 2, :, :], label_batch[:, :, :, :])
            bce_dice_loss = (dice_loss1 + dice_loss2) / 2
            loss = bce_dice_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            total_loss += loss
        avg_loss = total_loss/len(train_loader)
        logging.info('avg loss : %f' % (avg_loss))

        save_interval = 50  # int(max_epoch/6)

        if old_loss > avg_loss:
            old_loss = avg_loss
            # save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            save_mode_path = os.path.join(snapshot_path, 'epoch_best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            logging.info("save model!")

    writer.close()
    return "Training Finished!"



# def trainer_synapse(args, model, snapshot_path):
#     from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu
#     # max_iterations = args.max_iterations
#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
#                                transform=transforms.Compose(
#                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
#     print("The length of train set is: {}".format(len(db_train)))
#
#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)
#
#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
#                              worker_init_fn=worker_init_fn)
#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)
#     model.train()
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#     logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
#     best_performance = 0.0
#     iterator = tqdm(range(max_epoch), ncols=70)
#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#             outputs = model(image_batch)
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss = 0.4 * loss_ce + 0.6 * loss_dice
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_
#
#             iter_num = iter_num + 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)
#
#             logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
#
#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)
#
#         save_interval = 50  # int(max_epoch/6)
#         if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#
#         if epoch_num >= max_epoch - 1:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#             iterator.close()
#             break
#
#     writer.close()
#     return "Training Finished!"
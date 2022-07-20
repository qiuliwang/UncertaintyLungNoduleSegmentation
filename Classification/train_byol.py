from model.Models import *
from dataprocess import *
import loss.losses as losses
from metrics import *
import torch.optim as optim
import time
import numpy as np
import os
import torch
from config import Config
import shutil
from tqdm import tqdm
import imageio
import math
from bisect import bisect_right
import cv2

from model.byol import byol


config = Config()
model = byol(config, load = True)

torch.cuda.set_device(config.gpu)  

model_name = config.arch
if not os.path.isdir('result'):
    os.mkdir('result')
if config.resume is False:
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.seek(0)
        f.truncate()

model.cuda()

best_dice = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))    
dataloader, dataloader_val = get_dataloader(config, batchsize = config.batch_size)   # 64
criterion = losses.init_loss('BCE_logit').cuda()

if config.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    if config.evaluate:
        checkpoint = torch.load('./checkpoint/' + str(model_name) + '_best.pth.tar')
    else:
        checkpoint = torch.load('./checkpoint/' + str(model_name) + '.pth.tar')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_dice = checkpoint['dice']
    start_epoch = config.epochs

def adjust_lr(optimizer, epoch, eta_max=0.0001, eta_min=0.00005):
    cur_lr = 0.
    if config.lr_type == 'SGDR':
        i = int(math.log2(epoch / config.sgdr_t + 1))
        T_cur = epoch - config.sgdr_t * (2 ** (i) - 1)
        T_i = (config.sgdr_t * 2 ** i)

        cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

    elif config.lr_type == 'multistep':
        cur_lr = config.learning_rate * 0.1 ** bisect_right(config.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

def train(epoch):
    print('Training Epoch: ', epoch)
    model.train()
    train_loss = 0

    start_time = time.time()
    lr = adjust_lr(optimizer, epoch)
    for batch_idx, (inputs, targets_s, label) in tqdm(enumerate(dataloader)):
        iter_start_time = time.time()
        inputs = inputs.cuda()
        targets_s = targets_s.cuda()
        labels = label.cuda()

        outputs = model(inputs)

        outputs = torch.sigmoid(outputs)
        loss_seg = criterion(outputs, labels)
        loss_all = loss_seg

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        train_loss += loss_all.item()

    print('Epoch:{0}\t duration:{1:.3f}\ttrain_loss:{2:.6f}'.format(epoch, time.time()-start_time, train_loss/len(dataloader)))
    
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.write('Epoch:{0},duration:{1:.3f},learning_rate:{2:.6f},train_loss:{3:.4f}, '
          .format(epoch, time.time()-start_time, lr, train_loss/len(dataloader)))

def test(epoch):
    print('Testing Epoch: ', epoch)
    global best_dice
    model.eval()
    dices_all_s = []
    ious_all_s = []
    nsds_all_s = []

    with torch.no_grad():
        acc = 0
        for batch_idx, (inputs, targets_s, label) in enumerate(dataloader_val):
            
            inputs = inputs.cuda()
            targets_s = targets_s.cuda()
            label = label.cuda()
            outputs = model(inputs)
            logits = torch.sigmoid(outputs)

            pred = logits.argmax(dim = 1)
            label = label.argmax(dim = 1)

            num_correct = float(torch.eq(pred, label).sum().float().item())
            acc_ = num_correct / label.size()[0]
            acc += acc_
        acc = acc / len(dataloader_val)
        # print('num_correct: ', num_correct)
        # print('len(dataloader_val): ', len(dataloader_val))

        print('Epoch:{}\tbatch_idx:{}/All_batch:{}\tacc:{:.4f}'.format(epoch, batch_idx, len(dataloader_val), np.mean(np.array(acc))))

        with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
            f.write('Epoch:{}\tbatch_idx:{}/All_batch:{}\tacc:{:.4f}'.format(epoch, batch_idx, len(dataloader_val), np.mean(np.array(acc)))+ '\n')
    # Save checkpoint.
    # if config.resume is False:
    #     dice = np.mean(np.array(dices_all_s))
    #     print('Test accuracy: ', dice)
    #     state = {
    #         'model': model.state_dict(),
    #         'dice': dice,
    #         'epoch': epoch,
    #         'optimizer': optimizer.state_dict()
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/'+str(model_name)+'.pth.tar')

    #     is_best = False
    #     if best_dice < dice:
    #         best_dice = dice
    #         is_best = True

    #     if is_best:
    #         shutil.copyfile('./checkpoint/' + str(model_name) + '.pth.tar',
    #                         './checkpoint/' + str(model_name) + '_best.pth.tar')
    #     print('Save Successfully')
    #     print('------------------------------------------------------------------------')

if __name__ == '__main__':
    if config.resume:
        test(start_epoch)
    else:
        for epoch in tqdm(range(start_epoch, config.epochs)):
            train(epoch)
            test(epoch)

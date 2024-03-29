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

config = Config()

torch.cuda.set_device(config.gpu)  

model_name = config.arch
if not os.path.isdir('result'):
    os.mkdir('result')
if config.resume is False:
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.seek(0)
        f.truncate()

# model = U_Net(3,1)
model = AttU_Net(3, 1)
# model = R2U_Net ()

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

def adjust_lr(optimizer, epoch, eta_max=0.0001, eta_min=0.):
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
    for batch_idx, (inputs, targets_s) in tqdm(enumerate(dataloader)):
        iter_start_time = time.time()
        inputs = inputs.cuda()
        targets_s = targets_s.cuda()
        outputs = model(inputs)

        outputs_s_sig = torch.sigmoid(outputs)
        loss_seg = criterion(outputs_s_sig, targets_s)
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
        for batch_idx, (inputs, targets_s) in enumerate(dataloader_val):
            inputs = inputs.cuda()
            targets_s = targets_s.cuda()

            outputs = model(inputs)
            outputs_final_sig = torch.sigmoid(outputs)

            dices_all_s = meandice(outputs_final_sig, targets_s, dices_all_s)
            ious_all_s = meandIoU(outputs_final_sig, targets_s, ious_all_s)
            nsds_all_s = meanNSD(outputs_final_sig, targets_s, nsds_all_s)

            # saving image
            if config.evaluate and (epoch == 74 or epoch == 149 or epoch == 0):
                basePath = os.path.join(config.figurePath, str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '/fold' + str(config.fold))
                inputsPath = os.path.join(basePath, str(epoch), 'inputs')
                masksPath_s = os.path.join(basePath, str(epoch), 'masks_s')
                outputsPath_s = os.path.join(basePath, str(epoch), 'outputs_s')
                if not os.path.exists(inputsPath):
                    os.makedirs(inputsPath)
                if not os.path.exists(masksPath_s):
                    os.makedirs(masksPath_s)
                if not os.path.exists(outputsPath_s):
                    os.makedirs(outputsPath_s)

                num = inputs.shape[0]
                inputsfolder = inputs.chunk(num, dim=0)
                masksfolder_s = targets_s.chunk(num, dim=0)
                outputsfolder_s = outputs_final_sig.chunk(num, dim=0)
                for index in range(num):
                    input = inputsfolder[index]
                    input = input.squeeze()
                    imageio.imsave(os.path.join(inputsPath, str(epoch) + '_' + str(batch_idx*config.batch_size+index+1) + '.jpg'), input.cpu().detach().numpy().transpose(1,2,0))

                    target_s = masksfolder_s[index]
                    target_s = target_s.squeeze()
                    imageio.imsave(os.path.join(masksPath_s, str(epoch) + '_' + str(batch_idx*config.batch_size+index+1) + '_mask.jpg'), target_s.cpu().detach().numpy())

                    output_s = outputsfolder_s[index]
                    output_s = output_s.squeeze()
                    imageio.imsave(os.path.join(outputsPath_s, str(epoch) + '_' + str(batch_idx*config.batch_size+index+1) + '_pre.jpg'), output_s.cpu().detach().numpy())

        print('Epoch:{}\tbatch_idx:{}/All_batch:{}\tdice_s:{:.4f}\tiou_s:{:.4f}\tnsd_s:{:.4f}'.format(epoch, batch_idx, len(dataloader_val), np.mean(np.array(dices_all_s)), np.mean(np.array(ious_all_s)), np.mean(np.array(nsds_all_s))))
        with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
            f.write('Epoch:{},batch_idx:{}/All_batch:{},dice_s:{:.4f},iou_s:{:.4f},nsd_s:{:.4f}'
            .format(epoch, batch_idx, len(dataloader_val), np.mean(np.array(dices_all_s)), np.mean(np.array(ious_all_s)), np.mean(np.array(nsds_all_s)))+ '\n')

    # Save checkpoint.
    if config.resume is False:
        dice = np.mean(np.array(dices_all_s))
        print('Test accuracy: ', dice)
        state = {
            'model': model.state_dict(),
            'dice': dice,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+str(model_name)+'.pth.tar')

        is_best = False
        if best_dice < dice:
            best_dice = dice
            is_best = True

        if is_best:
            shutil.copyfile('./checkpoint/' + str(model_name) + '.pth.tar',
                            './checkpoint/' + str(model_name) + '_best.pth.tar')
        print('Save Successfully')
        print('------------------------------------------------------------------------')

if __name__ == '__main__':

    if config.resume:
        test(start_epoch)
    else:
        for epoch in tqdm(range(start_epoch, config.epochs)):
            train(epoch)
            test(epoch)

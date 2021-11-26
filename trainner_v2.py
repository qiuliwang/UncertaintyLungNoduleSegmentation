from model.Unet import *
from model.Vggnet import VGG16Net
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

config = Config()

torch.cuda.set_device(config.gpu)  

model_name = config.arch
if not os.path.isdir('result'):
    os.mkdir('result')
if config.evaluate is False:
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.seek(0)
        f.truncate()
model = VGG16Net(num_classes=config.out_dim)
# model = Unet(3, 1)
model.cuda()
best_dice = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))    
dataloader, dataloader_val = get_dataloader(config, batchsize=64)   # 64
criterion = losses.init_loss('BCE_logit').cuda()
# criterion = losses.init_loss('DiceLoss').cuda()
criterion_im = losses.init_loss('IMLoss').cuda()
criterion_c = losses.init_loss('ContrastiveLoss').cuda()
criterion_kd = losses.init_loss('KDChannel').cuda()

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
    model.train()
    train_loss = 0

    start_time = time.time()
    lr = adjust_lr(optimizer, epoch)
    for batch_idx, (inputs, targets_i, targets_u, targets_d, targets_s) in enumerate(dataloader):
        iter_start_time = time.time()
        inputs = inputs.cuda()
        targets_i = targets_i.cuda()
        targets_u = targets_u.cuda()
        targets_s = targets_s.cuda()

        outputs = model(inputs, [])
        
        targets_i_sig = torch.sigmoid(targets_i)
        targets_u_sig = torch.sigmoid(targets_u)
        targets_s_sig = torch.sigmoid(targets_s)
        outputs_i_sig = torch.sigmoid(outputs[0])
        outputs_u_sig = torch.sigmoid(outputs[1])
        outputs_s_sig = torch.sigmoid(outputs[2])
        loss_seg_i = criterion(outputs_i_sig, targets_i_sig)
        loss_seg_u = criterion(outputs_u_sig, targets_u_sig)
        loss_seg_s = criterion(outputs_s_sig, targets_s_sig)

        outputs_coase, coase_logits = model(inputs, [], flag1 = True, flag2 = False)
        outputs_mask, mask_logits = model(inputs, [targets_i_sig, targets_u_sig], flag1 = True, flag2 = False)
        
        # # caculate importance map loss
        # losslist_im = []
        # for i in range(6):
        #     loss_i  = criterion_im(outputs_coase[i], outputs_mask[i])
        #     losslist_im.append(loss_i)
        # loss_im = sum(losslist_im)

        # caculate kd loss
        losslist_kd = []
        gt = torch.cat((targets_i_sig, targets_u_sig), dim=1)
        gt = torch.cat((gt, targets_s_sig), dim=1)
        for j in range(config.num_classes):
            loss_j  = criterion_kd(coase_logits[j], mask_logits[j], gt, config.num_classes)
            losslist_kd.append(loss_j)
        loss_kd = sum(losslist_kd)

        # loss_all = config.weight_seg*(loss_seg_i + loss_seg_u + loss_seg_s) + config.weight_im * loss_im + config.weight_kd * loss_kd
        loss_all = config.weight_seg*(loss_seg_i + loss_seg_u + loss_seg_s) + config.weight_kd * loss_kd

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        train_loss += loss_all.item()

        # print('Epoch:{}\t batch_idx:{}/All_batch:{}\t duration:{:.3f}\t loss_seg:{:.3f}\t loss_con:{:.3f}\t loss_all:{:.3f}'
        #   .format(epoch, batch_idx, len(dataloader), time.time()-iter_start_time, loss_seg.item(), loss_con.item(), loss_all.item()))
        print('Epoch:{}\t batch_idx:{}/All_batch:{}\t duration:{:.3f}\t loss_all:{:.3f}'
          .format(epoch, batch_idx, len(dataloader), time.time()-iter_start_time, loss_all.item()))
        iter_start_time = time.time()
    print('Epoch:{0}\t duration:{1:.3f}\ttrain_loss:{2:.6f}'.format(epoch, time.time()-start_time, train_loss/len(dataloader)))
    
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.write('Epoch:{0}\t duration:{1:.3f}\t learning_rate:{2:.6f}\t train_loss:{3:.6f}'
          .format(epoch, time.time()-start_time, lr, train_loss/len(dataloader)))

def test(epoch):
    global best_dice
    model.eval()
    dices_all_i = []
    dices_all_u = []
    dices_all_s = []
    dices_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets_i, targets_u, targets_d, targets_s) in enumerate(dataloader_val):
            inputs = inputs.cuda()
            targets_i = targets_i.cuda()
            targets_u = targets_u.cuda()
            targets_s = targets_s.cuda()
            targets_i_sig = torch.sigmoid(targets_i)
            targets_u_sig = torch.sigmoid(targets_u)
            targets_s_sig = torch.sigmoid(targets_s)
            
            outputs = model(inputs, [])

            outputs_i_sig = torch.sigmoid(outputs[0])
            outputs_u_sig = torch.sigmoid(outputs[1])
            outputs_s_sig = torch.sigmoid(outputs[2])

            dices_all_i = meandice(outputs_i_sig, targets_i_sig, dices_all_i)
            dices_all_u = meandice(outputs_u_sig, targets_u_sig, dices_all_u)
            dices_all_s = meandice(outputs_s_sig, targets_s_sig, dices_all_s)
            
            uncertainty_map = outputs_i_sig + outputs_u_sig + outputs_s_sig

            dices_all.append((np.mean(np.array(dices_all_i))+np.mean(np.array(dices_all_u))+np.mean(np.array(dices_all_s)))/3)

            # 保存图像
            if(epoch%99==0 or epoch == config.epochs):
                basePath = os.path.join(config.figurePath, str(os.path.basename(__file__).split('.')[0]) + '_' + model_name)
                inputsPath = os.path.join(basePath, 'inputs')
                masksPath_i = os.path.join(basePath, 'masks_i')
                masksPath_u = os.path.join(basePath, 'masks_u')
                masksPath_s = os.path.join(basePath, 'masks_s')
                outputsPath_i = os.path.join(basePath, 'outputs_i')
                outputsPath_u = os.path.join(basePath, 'outputs_u')
                outputsPath_s = os.path.join(basePath, 'outputs_s')
                uncertaintyPath = os.path.join(basePath, 'uncertainty_map')
                if not os.path.exists(inputsPath):
                    os.makedirs(inputsPath)
                if not os.path.exists(masksPath_i):
                    os.makedirs(masksPath_i)
                if not os.path.exists(masksPath_u):
                    os.makedirs(masksPath_u)
                if not os.path.exists(masksPath_s):
                    os.makedirs(masksPath_s)
                if not os.path.exists(outputsPath_i):
                    os.makedirs(outputsPath_i)
                if not os.path.exists(outputsPath_u):
                    os.makedirs(outputsPath_u)
                if not os.path.exists(outputsPath_s):
                    os.makedirs(outputsPath_s)
                if not os.path.exists(uncertaintyPath):
                    os.makedirs(uncertaintyPath)
                num = inputs.shape[0]
                inputsfolder = inputs.chunk(num, dim=0)
                masksfolder_i = targets_i_sig.chunk(num, dim=0)
                masksfolder_u = targets_u_sig.chunk(num, dim=0)
                masksfolder_s = targets_s_sig.chunk(num, dim=0)
                outputsfolder_i = outputs_i_sig.chunk(num, dim=0)
                outputsfolder_u = outputs_u_sig.chunk(num, dim=0)
                outputsfolder_s = outputs_s_sig.chunk(num, dim=0)
                uncertaintyfolder = uncertainty_map.chunk(num, dim=0)
                for index in range(num):
                    input = inputsfolder[index]
                    input = input.squeeze()
                    imageio.imsave(os.path.join(inputsPath, str(epoch) + '_' + str(batch_idx*64+index+1) + '.jpg'), input.cpu().detach().numpy())

                    target_i = masksfolder_i[index]
                    target_i = target_i.squeeze()
                    imageio.imsave(os.path.join(masksPath_i, str(epoch) + '_' + str(batch_idx*64+index+1) + '.jpg'), target_i.cpu().detach().numpy())

                    target_u = masksfolder_u[index]
                    target_u = target_u.squeeze()
                    imageio.imsave(os.path.join(masksPath_u, str(epoch) + '_' + str(batch_idx*64+index+1) + '.jpg'), target_u.cpu().detach().numpy())

                    target_s = masksfolder_s[index]
                    target_s = target_s.squeeze()
                    imageio.imsave(os.path.join(masksPath_s, str(epoch) + '_' + str(batch_idx*64+index+1) + '.jpg'), target_s.cpu().detach().numpy())

                    output_i = outputsfolder_i[index]
                    output_i = output_i.squeeze()
                    imageio.imsave(os.path.join(outputsPath_i, str(epoch) + '_' + str(batch_idx*64+index+1) + '.jpg'), output_i.cpu().detach().numpy())

                    output_u = outputsfolder_u[index]
                    output_u = output_u.squeeze()
                    imageio.imsave(os.path.join(outputsPath_u, str(epoch) + '_' + str(batch_idx*64+index+1) + '.jpg'), output_u.cpu().detach().numpy())

                    output_s = outputsfolder_s[index]
                    output_s = output_s.squeeze()
                    imageio.imsave(os.path.join(outputsPath_s, str(epoch) + '_' + str(batch_idx*64+index+1) + '.jpg'), output_s.cpu().detach().numpy())

                    map = uncertaintyfolder[index]
                    map = map.squeeze()
                    imageio.imsave(os.path.join(uncertaintyPath, str(epoch) + '_' + str(batch_idx*64+index+1) + '.jpg'), map.cpu().detach().numpy())

            print('Epoch:{}\t batch_idx:{}/All_batch:{}\t dice_i:{:.4f}\t dice_u:{:.4f}\t dice_s:{:.4f}\t dice_all:{:.4f}'
            .format(epoch, batch_idx, len(dataloader_val), np.mean(np.array(dices_all_i)), np.mean(np.array(dices_all_u)), np.mean(np.array(dices_all_s)), np.mean(np.array(dices_all))))
        with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
            f.write('\t dice_i:{:.4f}\t dice_u:{:.4f}\t dice_s:{:.4f}\t dice_all:{:.4f}'.format(np.mean(np.array(dices_all_i)), np.mean(np.array(dices_all_u)), np.mean(np.array(dices_all_s)), np.mean(np.array(dices_all)))+'\n')

    # Save checkpoint.
    if config.evaluate is False:
        dice = np.mean(np.array(dices_all))
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

    if config.evaluate:
        test(start_epoch)
    else:
        for epoch in tqdm(range(start_epoch, config.epochs)):
            train(epoch)
            test(epoch)

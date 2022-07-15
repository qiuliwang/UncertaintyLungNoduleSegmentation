import numpy as np
import torch


def testdice(pred, gt, dices, dice_now):
    """
    :return save img' dice value in IoUs
    """
    # dices = []
    # dices = []
    # print(pred.shape)
    # pred = pred.squeeze(0)
    # pred = np.transpose(pred.cpu().detach().numpy(), (1, 2, 0))
    # _, pred_mask = cv2.threshold(pred, 210, 255, cv2.THRESH_BINARY)
    # pre_mask = cv2.blur(pre_mask,(3,3))
    # pre_mask = cv2.blur(pre_mask,(3,3))
    pred = np.transpose(pred, (2, 0, 1))
    pred = torch.from_numpy(pred)
    pred = pred.unsqueeze(0)
    gt = np.transpose(gt, (2, 0, 1))
    gt = torch.from_numpy(gt)
    gt = gt.unsqueeze(0)
    # # print(pred_torch[0][0][48][48])
    # # print(gt_torch[0][0][48][48])
    pred[pred < 125] = 0
    pred[pred >= 125] = 1
    gt[gt < 125] = 0
    gt[gt >= 125] = 1
    pred_np = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()

    for x in range(pred.size()[0]):
        dice = np.sum(pred_np[x][gt[x] == 1]) * 2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
        dice_now = dice
        dices.append(dice)
    return dices, dice_now

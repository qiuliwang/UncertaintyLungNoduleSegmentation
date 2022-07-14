import numpy as np
import torch


def meandIoU(pred, gt, IoUs):
    """
    :return save img' IoU value in IoUs
    IoU 就是 Jaccard 指标
    """
    # IoUs = []
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    gt[gt < 0.5] = 0
    gt[gt >= 0.5] = 1
    pred = pred.type(torch.LongTensor)
    pred_np = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    for x in range(pred.size()[0]):
        IoU = np.sum(pred_np[x][gt[x] == 1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x] == 1]))
        IoUs.append(IoU)
    return IoUs

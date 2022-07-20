import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import torch.optim as optim
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import tqdm
from byol_pytorch import BYOL
import pytorch_lightning as pl
import random
# test model, a resnet 50
from U_Net import U_Net, U_Net_Down

resnet = U_Net_Down()
# arguments
resnet = models.resnet50(pretrained=True)

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 256
EPOCHS     = 501
LR         = 1e-4
NUM_GPUS   = 2
IMAGE_SIZE = 512
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = 8
resume = False
save_model = True
Flag = 1 #'1~10'
# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []
        i = 0
        for path in Path(f'{folder}').glob('**/*'):
            t = i % 10
            if t == Flag:
                _, ext = os.path.splitext(path)
                if ext.lower() in IMAGE_EXTS:
                    self.paths.append(path)
            i += 1

        print(f'{len(self.paths)} images found')
        random.shuffle(self.paths)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        # img = img.convert('RGB')
        return self.transform(img)

# main

ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

model = BYOL(
    resnet,
    image_size = IMAGE_SIZE,
    hidden_layer = 'avgpool',
    projection_size = 1024,
    projection_hidden_size = 4096,
    moving_average_decay = 0.99
)

# model = BYOL(
#     resnet,
#     image_size = 256,
#     hidden_layer = 'avgpool',
#     projection_size = 1024,
#     projection_hidden_size = 4096,
#     moving_average_decay = 0.99,
#     use_momentum = False       # turn off momentum in the target encoder
# )

if resume:
    print('Loading model: ', '/home1/qiuliwang/Code/byol-pytorch-master/examples/trad_torch/checkpoint/10.pth.tar')
    model.load_state_dict(torch.load('/home1/qiuliwang/Code/byol-pytorch-master/examples/trad_torch/checkpoint/10.pth.tar'), strict = False)
        
model.cuda()
model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))   

def train(epoch):
    print('Training:\n')
    model.train()
    train_loss = 0
    count = len(train_loader)
    for batch_idx, data in enumerate(tqdm.tqdm(train_loader)):
        # print(float(count))
        data = data.cuda()
        loss = model(data)
        # print(target_proj_one.shape)
        # print(target_proj_two.shape)
        loss_all = loss
        train_loss += loss.item()
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch:{0}\t train_loss:{1:.6f}'.format(epoch, loss_all))

    if save_model is True and epoch % 10 == 0:
        state = {
            'model': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        print('Saving Model')
        torch.save(state, './checkpoint/'+str(epoch)+'.pth.tar')

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        a = 1
        train(epoch)

    # trainer = pl.Trainer(
    #     gpus = NUM_GPUS,
    #     max_epochs = EPOCHS,
    #     accumulate_grad_batches = 1,
    #     sync_batchnorm = True
    # )

    # trainer.fit(model, train_loader)

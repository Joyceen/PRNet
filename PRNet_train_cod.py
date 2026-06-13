import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.PRNet import PRNet
from torchvision.utils import make_grid
from data_cod import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options_cod import opt


def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


# === Phase 1: Improved loss functions (from CGCOD) ===

def seg_loss(logits, mask):
    logits = F.interpolate(logits, size=mask.shape[2:], mode='bilinear', align_corners=False)
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class GaussianFilter2D(nn.Module):
    def __init__(self, window_size=11, in_channels=1, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.padding = window_size // 2
        self.sigma = sigma
        self.register_buffer(name="gaussian_window2d", tensor=self._get_gaussian_window2d())

    def _get_gaussian_window1d(self):
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, self.window_size, 1)

    def _get_gaussian_window2d(self):
        gaussian_window_1d = self._get_gaussian_window1d()
        w = torch.matmul(gaussian_window_1d, gaussian_window_1d.transpose(dim0=-1, dim1=-2))
        w.reshape(1, 1, self.window_size, self.window_size)
        return w.repeat(self.in_channels, 1, 1, 1)

    def forward(self, x):
        x = F.conv2d(input=x, weight=self.gaussian_window2d, padding=self.padding, groups=x.shape[1])
        return x


class SSIM(nn.Module):
    def __init__(self, window_size=11, in_channels=1, sigma=1.5, K1=0.01, K2=0.03, L=1):
        super().__init__()
        self.window_size = window_size
        self.C1 = (K1 * L) ** 2
        self.C2 = (K2 * L) ** 2
        self.gaussian_filter = GaussianFilter2D(window_size=window_size, in_channels=in_channels, sigma=sigma)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, y):
        mu_x = self.gaussian_filter(x)
        mu_y = self.gaussian_filter(y)
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y
        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x * mu_x + mu_y * mu_y + self.C1
        B2 = sigma2_x + sigma2_y + self.C2
        S = (A1 * A2) / (B1 * B2)
        return S.mean()


def l1_ssim_loss(logits, mask):
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    prob = logits.sigmoid()
    l1 = F.l1_loss(input=prob, target=mask, reduction="mean")
    ssim_obj = SSIM().to(device=logits.device)
    ssiml = 1 - ssim_obj(x=prob, y=mask)
    return l1 + ssiml


if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True
image_root = opt.rgb_root
gt_root = opt.gt_root


test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
save_path = opt.save_path

logging.basicConfig(filename=save_path + 'PRNet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("PRNet-Train_4_pairs")

model = PRNet()

num_parms = 0
if (opt.load is not None):
    model.load_pre(opt.load)
    print('load model from ', opt.load)


for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
USE_NEW_LOSS = False
# Phase 1 loss modes:
# 1 = seg_loss only (edge-weighted BCE + IoU)
# 2 = seg_loss + l1_ssim (Phase 1 original, too strong)
# 3 = original BCE + IoU + l1_ssim (add SSIM only)
LOSS_MODE = 0
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.cuda()
    model.train()

    sal_loss_all = 0
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            s1, s2, s3, s4 = model(images)

            CE = torch.nn.BCEWithLogitsLoss()
            if LOSS_MODE == 1:
                loss1 = seg_loss(s1, gts)
                loss2 = seg_loss(s2, gts)
                loss3 = seg_loss(s3, gts)
                loss4 = seg_loss(s4, gts)
            elif LOSS_MODE == 2:
                loss1 = seg_loss(s1, gts) + l1_ssim_loss(s1, gts)
                loss2 = seg_loss(s2, gts) + l1_ssim_loss(s2, gts)
                loss3 = seg_loss(s3, gts) + l1_ssim_loss(s3, gts)
                loss4 = seg_loss(s4, gts) + l1_ssim_loss(s4, gts)
            elif LOSS_MODE == 3:
                loss1 = CE(s1, gts) + iou_loss(s1, gts) + l1_ssim_loss(s1, gts)
                loss2 = CE(s2, gts) + iou_loss(s2, gts) + l1_ssim_loss(s2, gts)
                loss3 = CE(s3, gts) + iou_loss(s3, gts) + l1_ssim_loss(s3, gts)
                loss4 = CE(s4, gts) + iou_loss(s4, gts) + l1_ssim_loss(s4, gts)
            else:
                loss1 = CE(s1, gts) + iou_loss(s1, gts)
                loss2 = CE(s2, gts) + iou_loss(s2, gts)
                loss3 = CE(s3, gts) + iou_loss(s3, gts)
                loss4 = CE(s4, gts) + iou_loss(s4, gts)
            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} , mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data,memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s1[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'PRNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'PRNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

def bce2d_new(input, target, reduction=None):
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg
    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res,res2,res3,res4 = model(image)
            res = res+res2+res3+res4
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'PRNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)

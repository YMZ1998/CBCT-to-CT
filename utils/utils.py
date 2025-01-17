import logging
import math
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int):
    """
    设置平台无关的随机种子，确保实验结果的可重复性。
    :param seed: 随机种子
    """
    # 设置 Python 的哈希种子，确保字典等结构的可重复性
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 设置 PyTorch GPU 随机种子（如果使用CUDA）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU，这行代码也可以保证一致性

    # 设置 CuDNN 使用确定性算法，这对于 GPU 上的操作是必要的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动调优，以确保每次运行结果相同
    print(f'Seed set to {seed} for all libraries (Python, NumPy, PyTorch)')


def get_logger(log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timer = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
    txthandle = logging.FileHandler((log_path + '/' + timer + '_log.txt'))
    txthandle.setFormatter(formatter)
    logger.addHandler(txthandle)
    return logger


class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.register_buffer("kernel", self._cal_gaussian_kernel(11, 1.5))
        self.L = 2.0
        self.k1 = 0.01
        self.k2 = 0.03

    @staticmethod
    def _cal_gaussian_kernel(size, sigma):
        g = torch.Tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
        g = g / g.sum()
        window = g.reshape([-1, 1]).matmul(g.reshape([1, -1]))
        # kernel = torch.reshape(window, [1, 1, size, size]).repeat(3, 1, 1, 1)
        kernel = torch.reshape(window, [1, 1, size, size])
        return kernel

    def forward(self, img0, img1):
        """
        :param img0: range in (-1, 1)
        :param img1: range in (-1, 1)
        :return: SSIM loss i.e. 1 - ssim
        """
        mu0 = torch.nn.functional.conv2d(img0, self.kernel, padding=0, groups=1)
        mu1 = torch.nn.functional.conv2d(img1, self.kernel, padding=0, groups=1)
        mu0_sq = torch.pow(mu0, 2)
        mu1_sq = torch.pow(mu1, 2)
        var0 = torch.nn.functional.conv2d(img0 * img0, self.kernel, padding=0, groups=1) - mu0_sq
        var1 = torch.nn.functional.conv2d(img1 * img1, self.kernel, padding=0, groups=1) - mu1_sq
        covar = torch.nn.functional.conv2d(img0 * img1, self.kernel, padding=0, groups=1) - mu0 * mu1
        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2
        ssim_numerator = (2 * mu0 * mu1 + c1) * (2 * covar + c2)
        ssim_denominator = (mu0_sq + mu1_sq + c1) * (var0 + var1 + c2)
        ssim = ssim_numerator / ssim_denominator
        ssim_loss = 1.0 - ssim
        # print('ssim_loss', ssim_loss)
        return ssim_loss


class MixedPix2PixLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedPix2PixLoss, self).__init__()
        self.alpha = alpha
        self.ssim_loss = SSIMLoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred, target):
        ssim_loss = torch.mean(self.ssim_loss(pred, target))
        l1_loss = self.l1_loss(pred, target)
        weighted_mixed_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        return weighted_mixed_loss


class MixedPix2PixLoss_mask(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedPix2PixLoss_mask, self).__init__()
        self.alpha = alpha
        self.ssim_loss = SSIMLoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        print('loss: ssim-', alpha, 'l1-', 1 - alpha)

    def forward(self, pred, target, mask=None):
        if mask != None and torch.count_nonzero(mask).item() != 0:
            ssim_loss = self.ssim_loss(pred * mask, target * mask)
            ssim_loss = torch.sum(ssim_loss) / torch.count_nonzero(mask)

            l1_loss = self.l1_loss(pred * mask, target * mask)
            l1_loss = torch.sum(l1_loss) / torch.count_nonzero(mask)
        else:
            ssim_loss = torch.mean(self.ssim_loss(pred, target))
            l1_loss = torch.mean(self.l1_loss(pred, target))

        weighted_mixed_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        return weighted_mixed_loss

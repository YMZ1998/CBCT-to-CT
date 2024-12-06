import torch
import numpy as np
import os
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model_path, stage1, stage2, resbranch, optimizer_stage1, optimizer_stage2, optimizer_resbranch, criterion, device, logger, epoch_stage1, epoch_stage2, epoch_total):
        self.model_path = model_path
        self.stage1 = stage1
        self.stage2 = stage2
        self.resbranch = resbranch
        self.optimizer_stage1 = optimizer_stage1
        self.optimizer_stage2 = optimizer_stage2
        self.optimizer_resbranch = optimizer_resbranch
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.epoch_stage1 = epoch_stage1
        self.epoch_stage2 = epoch_stage2
        self.epoch_total = epoch_total

    def compute_loss(self, pred, target, mask, alpha=0.95):
        """计算加权的损失值"""
        return alpha * self.criterion(pred, target, mask) + (1 - alpha) * self.criterion(pred, target, 1 - mask)

    def save_model(self, epoch, loss):
        # 每 10 个 epoch 保存一次模型
        if epoch % 10 == 0:
            """保存模型和优化器状态"""
            torch.save({
                'epoch': epoch,
                'model_stage1': self.stage1.state_dict(),
                'model_stage2': self.stage2.state_dict(),
                'model_resbranch': self.resbranch.state_dict(),
                'loss': loss
            }, os.path.join(self.model_path, f'model_{epoch}.pth'))

        torch.save({
            'epoch': epoch,
            'model_stage1': self.stage1.state_dict(),
            'model_stage2': self.stage2.state_dict(),
            'model_resbranch': self.resbranch.state_dict(),
            'optimizer_stage1': self.optimizer_stage1.state_dict(),
            'optimizer_stage2': self.optimizer_stage2.state_dict(),
            'optimizer_resbranch': self.optimizer_resbranch.state_dict(),
            'loss': loss
        }, os.path.join(self.model_path, 'last.pth'))

        print(f'** save epoch {epoch}.')

    def train_one_epoch(self, data_loader_train, epoch, interval=1):
        loss_gbs = []
        self.stage1.train()
        self.stage2.train()
        self.resbranch.train()

        # 打印当前训练阶段
        if epoch <= self.epoch_stage1:
            print(f"stage first {epoch}/{self.epoch_stage1} epoch")
            current_stage = 1
        elif epoch <= self.epoch_stage2:
            print(f"stage second {epoch}/{self.epoch_stage2} epoch")
            current_stage = 2
        else:
            print(f"stage total {epoch}/{self.epoch_total} epoch")
            current_stage = 3

        for iteration, (images, _) in enumerate(tqdm(data_loader_train)):
            if iteration % interval != 0:
                continue

            images = images.to(self.device)
            origin_cbct, origin_ct, enhance_ct, mask = torch.split(images, [5, 1, 1, 1], dim=1)

            # Training Stage 1
            if current_stage == 1:
                self.optimizer_stage1.zero_grad()
                out_global = self.stage1(origin_cbct * mask)
                loss_gb = self.compute_loss(out_global, enhance_ct, mask)
                loss_gbs.append(loss_gb.item())
                loss_gb.backward()
                self.optimizer_stage1.step()

            # Training Stage 2
            elif current_stage == 2:
                self.optimizer_stage2.zero_grad()
                out_global = self.stage1(origin_cbct * mask)
                out_global_cp = out_global.clone().detach()
                out_second = self.stage2(out_global_cp * mask)
                loss_gb2 = self.compute_loss(out_second, origin_ct, mask)
                loss_gbs.append(loss_gb2.item())
                loss_gb2.backward()
                self.optimizer_stage2.step()

            # Total Training
            else:
                self.optimizer_stage1.zero_grad()
                self.optimizer_stage2.zero_grad()
                self.optimizer_resbranch.zero_grad()

                out_global = self.stage1(origin_cbct * mask)
                out_global_cp = out_global.clone().detach()
                out_second = self.stage2(out_global_cp * mask)
                out_third = self.resbranch(origin_cbct * mask)
                out = out_second + out_third

                loss_gb3 = self.compute_loss(out, origin_ct, mask)
                loss_gbs.append(loss_gb3.item())
                loss_gb3.backward()
                self.optimizer_resbranch.step()

        loss_gbs_v = np.sum(loss_gbs)
        print(f'train epoch: {epoch}, loss: {loss_gbs_v}')

        self.save_model(epoch, loss_gbs_v)

        # 记录日志
        log_str = f" train epoch: {epoch} loss_gb: {loss_gbs_v}"
        self.logger.info(log_str)

        return loss_gbs_v

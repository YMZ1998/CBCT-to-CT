import sys

import numpy as np
import torch
from tqdm import tqdm

from parse_args import get_best_weight_path, get_latest_weight_path


class ModelTrainer:
    def __init__(self, args, stage1, stage2, resbranch, optimizer_stage1, optimizer_stage2,
                 optimizer_resbranch,
                 lr_scheduler_stage1, lr_scheduler_stage2, lr_scheduler_resbranch, criterion, device, logger):
        self.stage1 = stage1
        self.stage2 = stage2
        self.resbranch = resbranch
        self.optimizer_stage1 = optimizer_stage1
        self.optimizer_stage2 = optimizer_stage2
        self.optimizer_resbranch = optimizer_resbranch
        self.lr_scheduler_stage1 = lr_scheduler_stage1
        self.lr_scheduler_stage2 = lr_scheduler_stage2
        self.lr_scheduler_resbranch = lr_scheduler_resbranch
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.epoch_stage1 = args.epoch_stage1
        self.epoch_stage2 = args.epoch_stage2
        self.epoch_total = args.epoch_total
        self.best_weight_path = get_best_weight_path(args, verbose=False)
        self.latest_weight_path = get_latest_weight_path(args, verbose=False)

    def compute_loss(self, pred, target, mask, alpha=0.95):
        """计算加权的损失值"""
        return alpha * self.criterion(pred, target, mask) + (1 - alpha) * self.criterion(pred, target, 1 - mask)

    def save_model(self, epoch, loss, weight_path: str):
        torch.save({
            'epoch': epoch,
            'model_stage1': self.stage1.state_dict(),
            'model_stage2': self.stage2.state_dict(),
            'model_resbranch': self.resbranch.state_dict(),
            'optimizer_stage1': self.optimizer_stage1.state_dict(),
            'optimizer_stage2': self.optimizer_stage2.state_dict(),
            'optimizer_resbranch': self.optimizer_resbranch.state_dict(),
            'lr_scheduler_stage1': self.lr_scheduler_stage1.state_dict(),
            'lr_scheduler_stage2': self.lr_scheduler_stage2.state_dict(),
            'lr_scheduler_resbranch': self.lr_scheduler_resbranch.state_dict(),
            'loss': loss
        }, weight_path)

    def save_epoch_model(self, epoch, loss):
        # 每 10 个 epoch 保存一次模型
        # if epoch % 10 == 0:
        #     """保存模型和优化器状态"""
        #     torch.save({
        #         'epoch': epoch,
        #         'model_stage1': self.stage1.state_dict(),
        #         'model_stage2': self.stage2.state_dict(),
        #         'model_resbranch': self.resbranch.state_dict(),
        #         'loss': loss
        #     }, os.path.join(self.model_path, f'model_{epoch}.pth'))
        self.save_model(epoch, loss, weight_path=self.latest_weight_path)

    def train_one_epoch(self, data_loader_train, epoch, interval=1):
        loss_gbs = []
        self.stage1.train()
        self.stage2.train()
        self.resbranch.train()

        print('-' * 20)
        if epoch <= self.epoch_stage1:
            print('stage 1 epoch {}/{} lr {:.6f}'.format(epoch, self.epoch_stage1,
                                                         self.optimizer_stage1.param_groups[0]["lr"]))
            current_stage = 1
        elif epoch <= self.epoch_stage2:
            print('stage 2 epoch {}/{} lr {:.6f}'.format(epoch, self.epoch_stage2,
                                                         self.optimizer_stage2.param_groups[0]["lr"]))
            current_stage = 2
        else:
            print('stage total epoch {}/{} lr {:.6f}'.format(epoch, self.epoch_total,
                                                             self.optimizer_resbranch.param_groups[0]["lr"]))
            current_stage = 3
        print('-' * 20)
        best_loss = 999
        data_loader_train = tqdm(data_loader_train, file=sys.stdout)
        for images, _ in data_loader_train:

            images = images.to(self.device)
            origin_cbct, origin_ct, enhance_ct, mask = torch.split(images, [5, 1, 1, 1], dim=1)
            mask.fill_(1.0)
            # Training Stage 1
            if current_stage == 1:
                self.optimizer_stage1.zero_grad()
                out_global = self.stage1(origin_cbct * mask)
                # print("out_global", torch.min(out_global).item(), torch.max(out_global).item())
                # print("enhance_ct", torch.min(enhance_ct).item(), torch.max(enhance_ct).item())
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
                # print("out_second", torch.min(out_second).item(), torch.max(out_second).item())
                # print("origin_ct", torch.min(origin_ct).item(), torch.max(origin_ct).item())
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
                # out = (out_second + out_third) / 2
                out = torch.tanh(out_second + out_third)

                loss_gb3 = self.compute_loss(out, origin_ct, mask)
                loss_gbs.append(loss_gb3.item())
                loss_gb3.backward()
                self.optimizer_resbranch.step()
            data_loader_train.desc = f"[train epoch {epoch}] loss: {np.mean(loss_gbs):.4f} "
        if current_stage == 1:
            self.lr_scheduler_stage1.step()
        elif current_stage == 2:
            self.lr_scheduler_stage2.step()
        else:
            self.lr_scheduler_resbranch.step()

        loss_gbs_v = np.mean(loss_gbs)
        # print(f'train epoch: {epoch}, loss: {loss_gbs_v}')

        self.save_epoch_model(epoch, loss_gbs_v)
        if current_stage == 3 and loss_gbs_v < best_loss:
            self.best_loss = loss_gbs_v
            self.save_model(epoch, loss_gbs_v, weight_path=self.best_weight_path)

        # 记录日志
        log_str = f" train epoch: {epoch} loss_gb: {loss_gbs_v}"
        self.logger.info(log_str)

        return loss_gbs_v


if __name__ == '__main__':
    from train import train

    train()

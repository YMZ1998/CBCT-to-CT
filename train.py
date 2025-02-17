import datetime
import math
import os
import time

import torch
import wandb
from torch import optim

from make_dataset import CreateDataset
from model_tester import ModelTester
from model_trainer import ModelTrainer
from parse_args import get_device, parse_args, check_dir, get_model, get_latest_weight_path
from utils.utils import get_logger, MixedPix2PixLoss_mask


def train():
    args = parse_args()
    check_dir(args)
    device = get_device()
    logger = get_logger(args.log_path)

    dataset_train_path = [os.path.join(args.dataset_path, p) for p in os.listdir(args.dataset_path) if
                          'train' in p and args.anatomy in p]
    dataset_test_path = [os.path.join(args.dataset_path, p) for p in os.listdir(args.dataset_path) if
                         'test' in p and args.anatomy in p][:]

    if args.wandb:
        assert wandb.run is None
        wandb.init(project=args.project_name, config=vars(args))
        assert wandb.run is not None
        print('Config:', wandb.config)

    stage1, stage2, resbranch = get_model(args)

    optimizer_stage1 = optim.AdamW(stage1.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer_stage2 = optim.AdamW(stage2.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer_resbranch = optim.AdamW(resbranch.parameters(), lr=args.learning_rate, weight_decay=0.01)

    lrf = 0.01
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epoch_total)) / 2) * (1 - lrf) + lrf  # cosine
    lr_scheduler_stage1 = torch.optim.lr_scheduler.LambdaLR(optimizer_stage1, lr_lambda=lf)
    lr_scheduler_stage2 = torch.optim.lr_scheduler.LambdaLR(optimizer_stage2, lr_lambda=lf)
    lr_scheduler_resbranch = torch.optim.lr_scheduler.LambdaLR(optimizer_resbranch, lr_lambda=lf)

    print('Training...')
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    dataset_train = CreateDataset(dataset_train_path)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=num_workers, pin_memory=True, drop_last=True)
    dataset_test = CreateDataset(dataset_test_path)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers,
                                                   pin_memory=True, drop_last=True)

    criterion = MixedPix2PixLoss_mask(alpha=0.5).to(device)

    model_trainer = ModelTrainer(
        args,
        stage1=stage1,
        stage2=stage2,
        resbranch=resbranch,
        optimizer_stage1=optimizer_stage1,
        optimizer_stage2=optimizer_stage2,
        optimizer_resbranch=optimizer_resbranch,
        lr_scheduler_stage1=lr_scheduler_stage1,
        lr_scheduler_stage2=lr_scheduler_stage2,
        lr_scheduler_resbranch=lr_scheduler_resbranch,
        criterion=criterion,
        device=device,
        logger=logger,
    )

    last_epoch = 0
    weight_path = get_latest_weight_path(args)
    if args.resume:
        if os.path.exists(weight_path):
            last_epoch = model_trainer.load_model_weights(weight_path)
        else:
            print('No checkpoint found, starting from scratch.')

    model_tester = ModelTester(stage1=stage1, stage2=stage2, resbranch=resbranch, device=device,
                               epoch_stage1=args.epoch_stage1, epoch_stage2=args.epoch_stage2, logger=logger)

    start_time = time.time()
    for epoch in range(last_epoch + 1, args.epoch_total + 1):
        train_loss = model_trainer.train_one_epoch(data_loader_train, epoch)
        if epoch % 3 == 0:
            test_metrics = model_tester.test(data_loader_test, epoch)
            if args.wandb:
                wandb.log({
                    "train loss": train_loss,
                    "psnr": test_metrics['psnr'],
                    "ssim": test_metrics['ssim'],
                    "mae": test_metrics['mae']
                })

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    train()

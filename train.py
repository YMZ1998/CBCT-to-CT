import wandb
from torch import optim

from dataset import *
from model_tester import ModelTester
from model_trainer import ModelTrainer
from parse_args import get_device, parse_args, check_dir, get_model
from utils import *


def load_checkpoint(checkpoint_path, stage1, stage2, resbranch, optimizer_stage1, optimizer_stage2,
                    optimizer_resbranch):
    checkpoint = torch.load(checkpoint_path)

    stage1.load_state_dict(checkpoint['model_stage1'])
    stage2.load_state_dict(checkpoint['model_stage2'])
    resbranch.load_state_dict(checkpoint['model_resbranch'])

    optimizer_stage1.load_state_dict(checkpoint['optimizer_stage1'])
    optimizer_stage2.load_state_dict(checkpoint['optimizer_stage2'])
    optimizer_resbranch.load_state_dict(checkpoint['optimizer_resbranch'])

    last_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {last_epoch} with loss: {last_loss}")
    return last_epoch


def train():
    args = parse_args()
    check_dir(args)
    device = get_device()
    logger = get_logger(args.log_path)

    # 设置数据集路径
    dataset_train_path = args.dataset_path[0] if args.anatomy == 'brain' else args.dataset_path[2]
    dataset_test_path = args.dataset_path[1] if args.anatomy == 'brain' else args.dataset_path[3]


    if args.wandb:
        assert wandb.run is None
        run = wandb.init(project=args.project_name, config=vars(args))
        assert wandb.run is not None
        print('Config:', wandb.config)

    stage1, stage2, resbranch = get_model(device)

    optimizer_stage1 = optim.AdamW(stage1.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer_stage2 = optim.AdamW(stage2.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer_resbranch = optim.AdamW(resbranch.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # 如果需要恢复训练
    last_epoch = 0
    if args.resume:
        if os.path.exists(args.checkpoint_path):
            last_epoch = load_checkpoint(args.checkpoint_path, stage1, stage2, resbranch,
                                                    optimizer_stage1,
                                                    optimizer_stage2, optimizer_resbranch)
        else:
            print('No checkpoint found, starting from scratch.')

    print('Training...')
    dataset_train = CreateDataset(dataset_path=dataset_train_path)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=4, pin_memory=True, drop_last=True)
    dataset_test = CreateDataset(dataset_path=dataset_test_path)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                   pin_memory=True, drop_last=True)

    # 损失函数
    criterion = MixedPix2PixLoss_mask(alpha=0.5).to(device)

    model_trainer = ModelTrainer(
        model_path=args.model_path,
        stage1=stage1,
        stage2=stage2,
        resbranch=resbranch,
        optimizer_stage1=optimizer_stage1,
        optimizer_stage2=optimizer_stage2,
        optimizer_resbranch=optimizer_resbranch,
        criterion=criterion,
        device=device,
        logger=logger,
        epoch_stage1=args.epoch_stage1,
        epoch_stage2=args.epoch_stage2,
        epoch_total=args.epoch_total
    )

    model_tester = ModelTester(stage1=stage1, stage2=stage2, resbranch=resbranch, device=device,
                               epoch_stage1=args.epoch_stage1, epoch_stage2=args.epoch_stage2, logger=logger)

    # 训练和测试循环
    for epoch in range(last_epoch + 1, args.epoch_total + 1):
        train_loss = model_trainer.train_one_epoch(data_loader_train, epoch, interval=2)
        test_metrics = model_tester.test(data_loader_test, epoch)
        if args.wandb:
            wandb.log({
                "train loss": train_loss,
                "psnr": test_metrics['psnr'],
                "ssim": test_metrics['ssim'],
                "mae": test_metrics['mae']
            })


if __name__ == '__main__':
    train()

import shutil

import wandb
from torch import optim

from dataset import *
from model_tester import ModelTester
from model_trainer import ModelTrainer
from network import *
from utils import *

set_seed_torch(14)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    return device


if __name__ == '__main__':

    config = {
        'anatomy': 'brain',  # 'brain' or 'pelvis
        'resume': False,
        'wandb': False,
        'iftrain': True,
        'iftest': True,
        'project_name': 'synthRAD_CBCT_to_CT',
        'epoch_stage1': 20,
        'epoch_stage2': 40,
        'epoch_total': 60,
        'batch_size': 2,
        'device_num': '0',
        'learning_rate': 0.0001,
        'model_path': 'checkpoint',
        'last_checkpoint_name': 'checkpoint/last.pth',
        'dataset_path': ['./dataset/synthRAD_interval_2_brain_train.npz',
                         './dataset/synthRAD_interval_1_brain_test.npz',
                         './dataset/synthRAD_interval_2_pelvis_train.npz',
                         './dataset/synthRAD_interval_1_pelvis_test.npz'],
        'log_path': 'log',
        'visual_path': 'visualization'
    }

    anatomy = config['anatomy']

    resume = config['resume']
    ifwandb = config['wandb']
    iftrain = config['iftrain']
    iftest = config['iftest']

    projectname = config['project_name']
    epoch_stage1 = config['epoch_stage1']
    epoch_stage2 = config['epoch_stage2']
    epoch_total = config['epoch_total']

    batch_size = config['batch_size']
    device = get_device()
    learning_rate = config['learning_rate']
    model_path = config['model_path']
    last_checkpoint_name = config['last_checkpoint_name']

    if anatomy == 'brain':
        dataset_train_path = config['dataset_path'][0]
        dataset_test_path = config['dataset_path'][1]
    else:
        dataset_train_path = config['dataset_path'][2]
        dataset_test_path = config['dataset_path'][3]

    log_path = config['log_path']
    visual_path = config['visual_path']
    last_epoch = 0
    last_loss = 0

    if ifwandb:
        assert wandb.run is None
        run = wandb.init(project=projectname, config=config)
        assert wandb.run is not None
        print('config:', wandb.config)

    stage1 = MyUNet_plus(32).to(device)
    stage2 = MyUNet(32).to(device)
    resbranch = MyUNet_plus(32, act=False).to(device)

    optimizer_stage1 = optim.AdamW(stage1.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer_stage2 = optim.AdamW(stage2.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer_resbranch = optim.AdamW(resbranch.parameters(), lr=learning_rate, weight_decay=0.01)

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = get_logger(log_path)

    if os.path.exists(visual_path):
        shutil.rmtree(visual_path)
    os.makedirs(visual_path, exist_ok=True)

    if resume:
        if not os.path.exists(last_checkpoint_name):
            print('no last checkpoint, start a new train.')
        else:
            checkpoint = torch.load(last_checkpoint_name)

            stage1.load_state_dict(checkpoint['model_stage1'])
            stage2.load_state_dict(checkpoint['model_stage2'])
            resbranch.load_state_dict(checkpoint['model_resbranch'])

            optimizer_stage1.load_state_dict(checkpoint['optimizer_stage1'])
            optimizer_stage2.load_state_dict(checkpoint['optimizer_stage2'])
            optimizer_resbranch.load_state_dict(checkpoint['optimizer_resbranch'])

            last_epoch = checkpoint['epoch']
            last_loss = checkpoint['loss']
            print('load checkpoint from epoch {} loss:{}'.format(last_epoch, last_loss))

    if iftrain:
        print('train...')
        dataset_train = CreateDataset_npz(dataset_path=dataset_train_path)
        data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        sampler=None,
                                                        drop_last=True)

    if iftest:
        print('test...')
        dataset_test = CreateDataset_npz(dataset_path=dataset_test_path)
        data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       pin_memory=True,
                                                       sampler=None,
                                                       drop_last=True)

    criterion = MixedPix2PixLoss_mask(alpha=0.5).to(device)

    # 实例化 ModelTrainer
    model_trainer = ModelTrainer(
        model_path=model_path,
        stage1=stage1,  # 第一阶段模型
        stage2=stage2,  # 第二阶段模型
        resbranch=resbranch,  # 分支模型
        optimizer_stage1=optimizer_stage1,  # 第一阶段优化器
        optimizer_stage2=optimizer_stage2,  # 第二阶段优化器
        optimizer_resbranch=optimizer_resbranch,  # 分支优化器
        criterion=criterion,  # 损失函数
        device=get_device(),  # 设备 (CPU 或 GPU)
        logger=logger,  # 日志记录器
        epoch_stage1=epoch_stage1,  # 第一阶段训练结束的 epoch
        epoch_stage2=epoch_stage2,  # 第二阶段训练结束的 epoch
        epoch_total=epoch_total  # 总的训练 epoch
    )
    model_tester = ModelTester(stage1=stage1,
                               stage2=stage2,
                               resbranch=resbranch,
                               device=device,
                               epoch_stage1=epoch_stage1,
                               epoch_stage2=epoch_stage2,
                               logger=logger)

    # 训练循环
    for epoch in range(last_epoch + 1, epoch_total + 1):
        if iftrain:
            train_loss = model_trainer.train_one_epoch(data_loader_train, epoch, interval=2)

            if iftest and epoch % 1 == 0:
                test_metrics = model_tester.test(data_loader_test, epoch)
                if ifwandb and iftrain and iftest:
                    wandb.log({"train loss": train_loss,
                               "psnr": test_metrics['psnr'],
                               "ssim": test_metrics['ssim'],
                               "mae": test_metrics['mae']
                               })

        if iftest and not iftrain:
            test_metrics = model_tester.test(data_loader_test, epoch)
            break

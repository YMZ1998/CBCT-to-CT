import argparse
import os
import shutil

import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    return device


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def check_dir(args):
    ensure_dir_exists(args.model_path)
    ensure_dir_exists(args.log_path)
    ensure_dir_exists(args.visual_path)
    if os.path.exists(args.visual_path):
        shutil.rmtree(args.visual_path)
    os.makedirs(args.visual_path, exist_ok=True)


def get_model(args):
    print('★' * 30)
    print(f'model:{args.arch}')
    print('★' * 30)
    device = get_device()
    if args.arch == 'unet':
        from network.unet import MyUNet_plus, MyUNet

        stage1 = MyUNet_plus(32).to(device)
        stage2 = MyUNet(32).to(device)
        resbranch = MyUNet_plus(32, act=False).to(device)
        return stage1, stage2, resbranch

    elif 'efficientnet' in args.arch:
        from network.efficientnet_unet import EfficientUNet
        stage1 = EfficientUNet(5, model_name=args.arch).to(device)
        stage2 = EfficientUNet(1, model_name=args.arch).to(device)
        resbranch = EfficientUNet(5, model_name=args.arch, act=True).to(device)
        return stage1, stage2, resbranch

    else:
        raise ValueError('arch error')


def parse_args():
    from utils import set_seed
    set_seed(3407)
    epoch_step = 50

    parser = argparse.ArgumentParser(description="Train or test the CBCT to CT model")
    # 添加命令行参数
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet_b1', help='unet//efficientnet_b0')
    parser.add_argument('--anatomy', choices=['brain', 'pelvis'], default='brain', help="The anatomy type")
    parser.add_argument('--resume', default=False, type=bool, help="Resume from the last checkpoint")
    parser.add_argument('--wandb', default=False, type=bool, help="Enable wandb logging")
    parser.add_argument('--project_name', type=str, default='synthRAD_CBCT_to_CT', help="Wandb project name")
    parser.add_argument('--epoch_stage1', type=int, default=epoch_step, help="Epoch count for stage 1")
    parser.add_argument('--epoch_stage2', type=int, default=epoch_step * 2, help="Epoch count for stage 2")
    parser.add_argument('--epoch_total', type=int, default=epoch_step * 3, help="Total epoch count")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--model_path', type=str, default='checkpoint', help="Path to save model checkpoints")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/last.pth',
                        help="Path to the last checkpoint")
    parser.add_argument('--dataset_path', nargs=4, type=str, default=[
        './dataset/synthRAD_interval_2_brain_train.npz',
        './dataset/synthRAD_interval_1_brain_test.npz',
        './dataset/synthRAD_interval_2_pelvis_train.npz',
        './dataset/synthRAD_interval_1_pelvis_test.npz'
    ], help="Paths to training and testing datasets")
    parser.add_argument('--log_path', type=str, default='log', help="Log path")
    parser.add_argument('--visual_path', type=str, default='visualization', help="Visualization path")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    from torchsummary import summary

    image_size = 320

    stage1, stage2, resbranch = get_model(args)
    # summary(stage1, (5, image_size, image_size))
    # summary(stage2, (1, image_size, image_size))
    # summary(resbranch, (5, image_size, image_size))

    device = get_device()


    class Model(torch.nn.Module):
        def __init__(self, stage1, stage2, resbranch):
            super(Model, self).__init__()
            self.stage1 = stage1
            self.stage2 = stage2
            self.resbranch = resbranch

        def forward(self, x):
            x1 = self.stage1(x)
            x2 = self.stage2(x1)
            x3 = self.resbranch(x)
            return x2 + x3


    model = Model(stage1, stage2, resbranch).to(device)
    summary(model, (5, image_size, image_size))

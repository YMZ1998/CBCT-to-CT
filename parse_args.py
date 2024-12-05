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


def parse_args():
    parser = argparse.ArgumentParser(description="Train or test the CBCT to CT model")

    # 添加命令行参数
    parser.add_argument('--anatomy', choices=['brain', 'pelvis'], default='brain', help="The anatomy type")
    parser.add_argument('--resume', action='store_true', help="Resume from the last checkpoint")
    parser.add_argument('--wandb', action='store_true', help="Enable wandb logging")
    parser.add_argument('--iftrain', action='store_true', help="Whether to train the model")
    parser.add_argument('--iftest', action='store_true', help="Whether to test the model")
    parser.add_argument('--project_name', type=str, default='synthRAD_CBCT_to_CT', help="Wandb project name")
    parser.add_argument('--epoch_stage1', type=int, default=20, help="Epoch count for stage 1")
    parser.add_argument('--epoch_stage2', type=int, default=40, help="Epoch count for stage 2")
    parser.add_argument('--epoch_total', type=int, default=60, help="Total epoch count")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--model_path', type=str, default='checkpoint', help="Path to save model checkpoints")
    parser.add_argument('--last_checkpoint_name', type=str, default='checkpoint/last.pth',
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

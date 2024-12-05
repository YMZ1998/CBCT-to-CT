from dataset import *
from model_tester import ModelTester
from network import *
from parse_args import get_device, parse_args, check_dir
from utils import *

set_seed_torch(14)


def test():
    args = parse_args()

    # 设置数据集路径
    dataset_test_path = args.dataset_path[1] if args.anatomy == 'brain' else args.dataset_path[3]

    device = get_device()

    # 初始化模型和优化器
    stage1 = MyUNet_plus(32).to(device)
    stage2 = MyUNet(32).to(device)
    resbranch = MyUNet_plus(32, act=False).to(device)

    check_dir(args)
    logger = get_logger(args.log_path)

    print('Loading checkpoint...')
    checkpoint = torch.load(args.checkpoint_path)
    stage1.load_state_dict(checkpoint['model_stage1'])
    stage2.load_state_dict(checkpoint['model_stage2'])
    resbranch.load_state_dict(checkpoint['model_resbranch'])

    print('Testing...')
    dataset_test = CreateDataset_npz(dataset_path=dataset_test_path)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                   pin_memory=True, drop_last=True)

    model_tester = ModelTester(stage1=stage1, stage2=stage2, resbranch=resbranch, device=device,
                               epoch_stage1=args.epoch_stage1, epoch_stage2=args.epoch_stage2, logger=logger)

    model_tester.test(data_loader_test, 0)


if __name__ == '__main__':
    test()

from dataset import *
from model_tester import ModelTester
from parse_args import get_device, parse_args, check_dir, get_model
from utils import *


def test():
    args = parse_args()
    check_dir(args)
    device = get_device()
    logger = get_logger(args.log_path)

    dataset_test_path = args.dataset_path[1] if args.anatomy == 'brain' else args.dataset_path[3]
    stage1, stage2, resbranch = get_model(device)

    print('Loading checkpoint...')
    checkpoint = torch.load(args.checkpoint_path)
    stage1.load_state_dict(checkpoint['model_stage1'])
    stage2.load_state_dict(checkpoint['model_stage2'])
    resbranch.load_state_dict(checkpoint['model_resbranch'])

    print('Testing...')
    dataset_test = CreateDataset(dataset_path=dataset_test_path)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                   pin_memory=True, drop_last=True)

    model_tester = ModelTester(stage1=stage1, stage2=stage2, resbranch=resbranch, device=device,
                               epoch_stage1=args.epoch_stage1, epoch_stage2=args.epoch_stage2, logger=logger,
                               save_all=True)

    model_tester.test(data_loader_test, args.epoch_total)


if __name__ == '__main__':
    test()

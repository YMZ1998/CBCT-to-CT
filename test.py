import datetime
import os
import time

import torch

from make_dataset import CreateDataset
from model_tester import ModelTester
from parse_args import get_device, parse_args, check_dir, get_model, get_latest_weight_path, remove_and_create_dir
from utils import get_logger


def test():
    args = parse_args()
    remove_and_create_dir(args.visual_path)
    device = get_device()
    logger = get_logger(args.log_path)

    dataset_test_path = [os.path.join(args.dataset_path, p) for p in os.listdir(args.dataset_path) if
                         'test' in p and args.anatomy in p][:]
    stage1, stage2, resbranch = get_model(args)

    # print('Loading checkpoint...')
    # weight_path = get_best_weight_path(args)
    # weight_path = get_latest_weight_path(args)
    # checkpoint = torch.load(weight_path, weights_only=False, map_location='cpu')
    # stage1.load_state_dict(checkpoint['model_stage1'])
    # stage2.load_state_dict(checkpoint['model_stage2'])
    # resbranch.load_state_dict(checkpoint['model_resbranch'])

    print('Testing...')
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    dataset_test = CreateDataset(dataset_test_path)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers,
                                                   pin_memory=True, drop_last=True)

    model_tester = ModelTester(stage1=stage1, stage2=stage2, resbranch=resbranch, device=device,
                               epoch_stage1=args.epoch_stage1, epoch_stage2=args.epoch_stage2, logger=logger,
                               save_all=True)

    print('Loading checkpoint...')
    weight_path = get_latest_weight_path(args)
    model_tester.load_model_weights(weight_path)

    start_time = time.time()
    model_tester.test(data_loader_test, args.epoch_total)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("test time {}".format(total_time_str))


if __name__ == '__main__':
    test()

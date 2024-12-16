import datetime

import SimpleITK as sitk
from torch.utils.data import TensorDataset

from make_dataset import img_normalize, img_padding, window_transform, generate_2_5d_slices
from model_tester import ModelTester
from parse_args import get_device, parse_args, get_model, get_latest_weight_path
from utils import *


def load_data(cbct_path, ct_path, mask_path, shape):
    cbct = sitk.ReadImage(cbct_path)
    cbct = sitk.GetArrayFromImage(cbct)
    print(cbct.shape)
    cbct = img_normalize(cbct, type='cbct')
    cbct_padded, img_location = img_padding(cbct, shape[0], shape[1], -1)

    ct = sitk.ReadImage(ct_path)
    ct = sitk.GetArrayFromImage(ct)
    ct_padded, _ = img_padding(ct, shape[0], shape[1], -1000)
    ct_padded = img_normalize(ct_padded, type='ct')

    enhance_ct_norm = window_transform(ct, 1000, 350)
    enhance_ct_padded, _ = img_padding(enhance_ct_norm, shape[0], shape[1])
    enhance_ct_padded = enhance_ct_padded * 2 - 1

    mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask)
    mask_padded, _ = img_padding(mask, shape[0], shape[1])
    return cbct_padded, img_location, ct_padded, enhance_ct_padded, mask_padded


def predict():
    args = parse_args()
    shape = [args.image_size, args.image_size]
    data_path = f'./data/{args.anatomy}/test'
    case_path = [os.path.join(data_path, p) for p in os.listdir(data_path)][0]
    print(case_path)
    cbct_path = os.path.join(case_path, 'cbct.nii.gz')
    ct_path = os.path.join(case_path, 'ct.nii.gz')
    mask_path = os.path.join(case_path, 'mask.nii.gz')

    cbct_padded, img_location, ct_padded, enhance_ct_padded, mask_padded = load_data(cbct_path, ct_path, mask_path,
                                                                                     shape)
    length = len(cbct_padded)

    cbct_vecs, ct_vecs, enhance_ct_vecs, mask_vecs, location_vecs = [], [], [], [], []
    for index in range(length):
        cbct_2_5d = generate_2_5d_slices(cbct_padded, index, length)
        cbct_vecs.append(cbct_2_5d)
        location_vecs.append(img_location)
        ct_vecs.append(ct_padded[index])
        enhance_ct_vecs.append(enhance_ct_padded[index])
        mask_vecs.append(mask_padded[index])
    cbct_batch = np.array(cbct_vecs[:])
    ct_batch = np.expand_dims(np.array(ct_vecs[:]), axis=1)
    enhance_ct_batch = np.expand_dims(np.array(enhance_ct_vecs[:]), axis=1)
    mask_batch = np.expand_dims(np.array(mask_vecs[:]), axis=1).astype(np.float32)
    locations_batch = np.concatenate(location_vecs[:], axis=0).astype(np.float32)

    images_batch = np.concatenate((cbct_batch, ct_batch, enhance_ct_batch, mask_batch), axis=1)
    print(images_batch.shape)
    dataset_test = TensorDataset(torch.from_numpy(images_batch), torch.from_numpy(locations_batch))

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                   pin_memory=True, drop_last=True)

    device = get_device()
    logger = get_logger(args.log_path)

    stage1, stage2, resbranch = get_model(args)

    model_tester = ModelTester(stage1=stage1, stage2=stage2, resbranch=resbranch, device=device,
                               epoch_stage1=args.epoch_stage1, epoch_stage2=args.epoch_stage2, logger=logger,
                               save_all=True)
    print('Loading checkpoint...')
    weight_path = get_latest_weight_path(args)
    model_tester.load_model_weights(weight_path)

    print('Predict...')

    start_time = time.time()
    model_tester.predict(data_loader_test)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("test time {}".format(total_time_str))


if __name__ == '__main__':
    from image_metrics import compute_val_metrics

    predict()
    compute_val_metrics()

import os
import random

import SimpleITK as sitk
import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, TensorDataset
from tqdm import tqdm

from parse_args import parse_args


def img_normalize(img, type='cbct'):
    if type == 'cbct':
        min_value = np.min(img)
        max_value = np.max(img)
        img = (img - min_value) / (max_value - min_value + 1e-8)
        img = img * 2 - 1
    elif type == 'ct':
        min_value = -1024
        max_value = 3000
        img = (img - min_value) / (max_value - min_value)
        img = np.clip(img, 0, 1)
        img = img * 2 - 1
    else:
        pass
    return img


def img_padding(img, x=288, y=288, v=0):
    h, w = img.shape[1], img.shape[2]

    padding_y = (y - h) // 2, (y - h) - (y - h) // 2
    padding_x = (x - w) // 2, (x - w) - (x - w) // 2

    padded_img = np.pad(img, ((0, 0), padding_y, padding_x), mode='constant', constant_values=v)

    img_location = np.array([padding_x[0], padding_y[0], w, h])
    img_location = np.expand_dims(img_location, 0)
    return padded_img, img_location


def window_transform(ct_array, window_width, window_center):
    minWindow = window_center - 0.5 * window_width
    new_img = np.clip((ct_array - minWindow) / window_width, 0, 1)
    return new_img


def generate_2_5d_slices(image, index, length):
    """
    生成 2.5D 的 5 层切片。
    :param image: 输入图像
    :param index: 当前切片索引
    :param length: 图像切片总数
    :return: 5 层切片组成的数组
    """
    indices = [
        # max(0, index - 2),
        max(0, index - 1),
        index,
        min(length - 1, index + 1),
        # min(length - 1, index + 2)
    ]
    return np.array([image[i] for i in indices])


def generate_train_test_dataset(path, padding, p='brain', t='train', interval=3, save_path='dataset'):
    """
    生成训练或测试数据集。
    :param path: 数据集根目录
    :param padding: 填充参数
    :param p: 数据集标识
    :param t: 数据集类型 ('train' 或 'test')
    :param interval: 采样间隔
    :param save_path: 保存路径
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset_dirs = [d for d in os.listdir(path) if d != '.DS_Store']
    random.shuffle(dataset_dirs)
    print('dataset dirs: ', dataset_dirs)

    cbct_vecs, ct_vecs, enhance_ct_vecs, mask_vecs, location_vecs = [], [], [], [], []
    batch_size = 256
    batch_idx = 0
    for dir in tqdm(dataset_dirs, total=len(dataset_dirs), desc="Processing cases"):

        case_path = os.path.join(path, dir)
        cbct_path = os.path.join(case_path, 'cbct.nii.gz')
        ct_path = os.path.join(case_path, 'ct.nii.gz')
        mask_path = os.path.join(case_path, 'mask.nii.gz')

        cbct = sitk.ReadImage(cbct_path)
        cbct = sitk.GetArrayFromImage(cbct)
        cbct = img_normalize(cbct, type='cbct')
        cbct_padded, img_location = img_padding(cbct, padding[0], padding[1], -1)

        ct = sitk.ReadImage(ct_path)
        ct = sitk.GetArrayFromImage(ct)
        ct_padded, _ = img_padding(ct, padding[0], padding[1], -1000)
        ct_padded = img_normalize(ct_padded, type='ct')

        enhance_ct_norm = window_transform(ct, 1000, 350)
        enhance_ct_padded, _ = img_padding(enhance_ct_norm, padding[0], padding[1])
        enhance_ct_padded = enhance_ct_padded * 2 - 1

        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)
        mask_padded, _ = img_padding(mask, padding[0], padding[1])

        length = len(cbct_padded)

        for index in range(length):
            if index % interval != 0:
                continue

            cbct_2_5d = generate_2_5d_slices(cbct_padded, index, length)
            cbct_vecs.append(cbct_2_5d)
            location_vecs.append(img_location)
            ct_vecs.append(ct_padded[index])
            enhance_ct_vecs.append(enhance_ct_padded[index])
            mask_vecs.append(mask_padded[index])

        if len(cbct_vecs) > batch_size:
            print(len(cbct_vecs))
            # 获取当前批次数据
            cbct_batch = np.array(cbct_vecs[:batch_size])
            ct_batch = np.expand_dims(np.array(ct_vecs[:batch_size]), axis=1)
            enhance_ct_batch = np.expand_dims(np.array(enhance_ct_vecs[:batch_size]), axis=1)
            mask_batch = np.expand_dims(np.array(mask_vecs[:batch_size]), axis=1).astype(np.float32)
            locations_batch = np.concatenate(location_vecs[:batch_size], axis=0).astype(np.float32)
            cbct_vecs = cbct_vecs[batch_size:]
            ct_vecs = ct_vecs[batch_size:]
            enhance_ct_vecs = enhance_ct_vecs[batch_size:]
            mask_vecs = mask_vecs[batch_size:]
            location_vecs = location_vecs[batch_size:]

            # 打印统计信息
            print(
                f"Shapes - CBCT: {cbct_batch.shape}, CT: {ct_batch.shape}, Enhanced CT: {enhance_ct_batch.shape}, Mask: {mask_batch.shape}, Location: {locations_batch.shape}")
            print(f"CBCT min: {cbct_batch.min()}, max: {cbct_batch.max()}")
            print(f"CT min: {ct_batch.min()}, max: {ct_batch.max()}")
            print(f"Enhanced CT min: {enhance_ct_batch.min()}, max: {enhance_ct_batch.max()}")
            print(f"Mask min: {mask_batch.min()}, max: {mask_batch.max()}")

            # 合并数据
            images_batch = np.concatenate((cbct_batch, ct_batch, enhance_ct_batch, mask_batch), axis=1)

            # 保存数据集到 .npz 文件
            dataset_name = os.path.join(save_path, f'synthRAD_interval_{interval}_{p}_{t}_batch_{batch_idx + 1}.npz')
            print(f"Saving batch {batch_idx + 1} to: {dataset_name}")
            np.savez_compressed(dataset_name, images=images_batch, locations=locations_batch)
            batch_idx = batch_idx + 1


def load_npz_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ds = np.load(dataset_path)

    if 'images' not in ds or 'locations' not in ds:
        raise KeyError(f"The dataset at {dataset_path} must contain 'images' and 'locations' arrays.")

    return ds['images'], ds['locations']


def CreateDataset(dataset_paths):
    all_datasets = []

    for dataset_path in dataset_paths:
        print("dataset path:", dataset_path)
        img, location = load_npz_data(dataset_path)

        dataset = TensorDataset(torch.from_numpy(img), torch.from_numpy(location))
        all_datasets.append(dataset)

    combined_dataset = ConcatDataset(all_datasets)
    return combined_dataset


def check_dataset(anatomy):
    images, image_locations = load_npz_data(f'./dataset/synthRAD_interval_1_{anatomy}_test_batch_1.npz')
    print("images shape:", images.shape)
    # origin_cbct, origin_ct, enhance_ct, mask = np.split(images, [5, 1, 1, 1], dim=1)
    origin_cbct, origin_ct, enhance_ct, mask = np.split(images[100], [5, 6, 7], axis=0)

    plt.figure(figsize=(9, 3), dpi=100, tight_layout=True)
    # 显示 CBCT 图像
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(origin_cbct[2], cmap="gray")
    plt.title("CBCT")

    # 显示原始 CT 图像
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(origin_ct[0], cmap="gray")
    plt.title("Original CT")

    # 显示增强后的 CT 图像
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(enhance_ct[0], cmap="gray")
    plt.title("Enhanced CT")

    plt.show()


def generate_dataset(anatomy, shape):
    save_path = './dataset'
    for p in os.listdir(save_path):
        if anatomy in os.path.join(save_path, p):
            os.remove(os.path.join(save_path, p))
    generate_train_test_dataset(f'./data/{anatomy}/train', padding=shape, p=anatomy, t='train', interval=2,
                                save_path=save_path)
    generate_train_test_dataset(f'./data/{anatomy}/test', padding=shape, p=anatomy, t='test', interval=1,
                                save_path=save_path)


if __name__ == '__main__':
    args = parse_args()
    generate_dataset(args.anatomy, [args.image_size, args.image_size])
    # check_dataset(args.anatomy)

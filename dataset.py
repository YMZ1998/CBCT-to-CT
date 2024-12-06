import os
import random

import SimpleITK as sitk
import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

MASK_POINT = []
MASK_AREA = []


def generate_mask(img_size, padding, val=False):
    mask = np.ones(img_size)
    # print('mask size: ', mask.shape)

    mask_size_w = random.randint(img_size[0] // 4, img_size[0] - 1)
    mask_size_h = random.randint(img_size[1] // 4, img_size[1] - 1)
    # mask_size = random.randint(img_size//16, img_size//4)

    if (mask_size_w == img_size[0] - 1 and mask_size_h == img_size[1] - 1) or val:
        mask = np.expand_dims(1 - mask, axis=0)
        mask, location = mypadding(mask, x=padding[0], y=padding[1], v=1)
        return mask

    c_x = random.randint(0, img_size[0] - 1)
    c_y = random.randint(0, img_size[1] - 1)

    box_l_x = max(c_x - mask_size_w // 2, 0)
    box_l_y = max(c_y - mask_size_h // 2, 0)
    box_r_x = min(c_x + mask_size_w // 2, img_size[0] - 1)
    box_r_y = min(c_y + mask_size_h // 2, img_size[1] - 1)

    mask[box_l_y:box_r_y, box_l_x:box_r_x] = 0
    mask = np.expand_dims(mask, axis=0)
    mask, location = mypadding(mask, x=padding[0], y=padding[1], v=1)

    MASK_POINT.append([c_x + location[0][0], c_y + location[0][1]])
    MASK_AREA.append([box_l_y, box_r_y, box_l_x, box_r_x])
    return mask


def getmasks(img_size, channels, padding, val=False):
    masks = []
    for i in range(channels):
        # print('channel: ', i)
        masks.append(generate_mask(img_size, padding, val))
    masks = np.concatenate(masks, axis=0)
    return masks


def normalize(img, type='cbct'):
    if type == 'cbct':
        min_value = np.min(img)
        max_value = np.max(img)
        img = (img - min_value) / (max_value - min_value)
        img = img * 2 - 1
        # img = np.clip(img, 0, 1)
    elif type == 'ct':
        # 后面试试直接 min max，不固定
        min_value = -1024
        max_value = 3000
        img = (img - min_value) / (max_value - min_value)
        img = np.clip(img, 0, 1)
        img = img * 2 - 1
        # img = np.clip(img, 0, 1)
    else:
        pass
    return img


def mypadding(img, x=288, y=288, v=0):
    # 获取图片的高度和宽度
    h, w = img.shape[1], img.shape[2]

    # 计算前后填充
    padding_y = (y - h) // 2, (y - h) - (y - h) // 2
    padding_x = (x - w) // 2, (x - w) - (x - w) // 2

    # 使用 np.pad 进行填充
    padded_img = np.pad(img, ((0, 0), padding_y, padding_x), mode='constant', constant_values=v)

    # 返回填充后的图片和原始图片的位置
    img_location = np.array([padding_x[0], padding_y[0], w, h])
    img_location = np.expand_dims(img_location, 0)
    return padded_img, img_location


def cal_min_max(path):
    paths = os.listdir(path)

    size_heights = []
    size_widths = []

    cbct_min_values = []
    cbct_max_values = []

    ct_min_values = []
    ct_max_values = []

    for p in paths:
        if p == '.DS_Store':
            continue

        path_temp = os.path.join(path, p)
        path_cbct = os.path.join(path_temp, 'cbct.nii.gz')
        path_ct = os.path.join(path_temp, 'ct.nii.gz')

        cbct = sitk.ReadImage(path_cbct)
        cbct = sitk.GetArrayFromImage(cbct)

        # print(cbct.shape)
        size_heights.append(cbct.shape[1])
        size_widths.append(cbct.shape[2])
        cbct_min_values.append(cbct.min())
        cbct_max_values.append(cbct.max())

        ct = sitk.ReadImage(path_ct)
        ct = sitk.GetArrayFromImage(ct)
        ct_min_values.append(ct.min())
        ct_max_values.append(ct.max())

    print('size')
    print('--height: {}-{}'.format(np.min(size_heights), np.max(size_heights)))
    print('--width: {}-{}'.format(np.min(size_widths), np.max(size_widths)))

    print('cbct')
    print('--value: {}-{}'.format(np.min(cbct_min_values), np.max(cbct_max_values)))

    print('ct')
    print('--value: {}-{}'.format(np.min(ct_min_values), np.max(ct_max_values)))


def cal_min_max_val(path):
    paths = os.listdir(path)

    size_heights = []
    size_widths = []

    cbct_min_values = []
    cbct_max_values = []

    for p in paths:
        if p == '.DS_Store':
            continue

        path_temp = os.path.join(path, p)
        path_cbct = os.path.join(path_temp, 'cbct.nii.gz')

        cbct = sitk.ReadImage(path_cbct)
        cbct = sitk.GetArrayFromImage(cbct)

        size_heights.append(cbct.shape[1])
        size_widths.append(cbct.shape[2])
        cbct_min_values.append(cbct.min())
        cbct_max_values.append(cbct.max())

    print('size')
    print('--height: {}-{}'.format(np.min(size_heights), np.max(size_heights)))
    print('--width: {}-{}'.format(np.min(size_widths), np.max(size_widths)))

    print('cbct')
    print('--value: {}-{}'.format(np.min(cbct_min_values), np.max(cbct_max_values)))


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
        max(0, index - 2),
        max(0, index - 1),
        index,
        min(length - 1, index + 1),
        min(length - 1, index + 2)
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

    cbct_vecs, ct_vecs, enhance_ct_vecs, mask_vecs, location_vecs = [], [], [], [], []

    for dir in tqdm(dataset_dirs, total=len(dataset_dirs), desc="Processing cases"):
        case_path = os.path.join(path, dir)
        cbct_path = os.path.join(case_path, 'cbct.nii.gz')
        ct_path = os.path.join(case_path, 'ct.nii.gz')
        mask_path = os.path.join(case_path, 'mask.nii.gz')

        cbct = sitk.ReadImage(cbct_path)
        cbct = sitk.GetArrayFromImage(cbct)
        cbct_padded, img_location = mypadding(cbct, padding[0], padding[1])
        cbct_padded = normalize(cbct_padded, type='cbct')

        ct = sitk.ReadImage(ct_path)
        ct = sitk.GetArrayFromImage(ct)
        ct_padded, _ = mypadding(ct, padding[0], padding[1], -1000)
        ct_padded = normalize(ct_padded, type='ct')

        enhance_ct_norm = window_transform(ct, 1000, 350)
        enhance_ct_padded, _ = mypadding(enhance_ct_norm, padding[0], padding[1])
        enhance_ct_padded = enhance_ct_padded * 2 - 1

        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)
        mask_padded, _ = mypadding(mask, padding[0], padding[1])

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

    # 转换为 NumPy 数组并调整维度
    cbct_np = np.array(cbct_vecs)
    ct_np = np.expand_dims(np.array(ct_vecs), axis=1)
    enhance_ct_np = np.expand_dims(np.array(enhance_ct_vecs), axis=1)
    mask_np = np.expand_dims(np.array(mask_vecs), axis=1).astype(np.float32)
    locations = np.concatenate(location_vecs, axis=0).astype(np.float32)

    # 打印统计信息
    print(
        f"Shapes - CBCT: {cbct_np.shape}, CT: {ct_np.shape}, Enhanced CT: {enhance_ct_np.shape}, Mask: {mask_np.shape}, Location: {locations.shape}")
    print(f"CBCT min: {cbct_np.min()}, max: {cbct_np.max()}")
    print(f"CT min: {ct_np.min()}, max: {ct_np.max()}")
    print(f"Enhanced CT min: {enhance_ct_np.min()}, max: {enhance_ct_np.max()}")
    print(f"Mask min: {mask_np.min()}, max: {mask_np.max()}")

    # 保存数据集
    dataset_name = os.path.join(save_path, f'synthRAD_interval_{interval}_{p}_{t}.npz')
    print(f"Saving dataset to: {dataset_name}")
    np.savez(dataset_name, images=np.concatenate((cbct_np, ct_np, enhance_ct_np, mask_np), axis=1), locations=locations)


def load_npz_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ds = np.load(dataset_path)

    if 'images' not in ds or 'locations' not in ds:
        raise KeyError(f"The dataset at {dataset_path} must contain 'images' and 'locations' arrays.")

    return ds['images'], ds['locations']


def CreateDataset(dataset_path):
    print("dataset path:", dataset_path)
    img, location = load_npz_data(dataset_path)
    print("img shape:", img.shape)
    print("location shape:", location.shape)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(img), torch.from_numpy(location))
    return dataset


def check_dataset():
    images, image_locations = load_npz_data('./dataset/synthRAD_interval_1_brain_test.npz')
    print("images shape:", images.shape)
    # origin_cbct, origin_ct, enhance_ct, mask = np.split(images, [5, 1, 1, 1], dim=1)
    origin_cbct, origin_ct, enhance_ct, mask = np.split(images[100], [5, 6, 7], axis=0)

    fig = plt.figure(figsize=(9, 3), dpi=100, tight_layout=True)
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


def generate_dataset():
    brain_shape = [320, 320]
    generate_train_test_dataset('./data/train/brain', padding=brain_shape, p='brain', t='train', interval=2,
                                save_path='./dataset')
    generate_train_test_dataset('./data/test/brain', padding=brain_shape, p='brain', t='test', interval=1,
                                save_path='./dataset')

    # generate_train_test_dataset('./data/train/pelvis', padding=[592, 416], p='pelvis', t='train', interval=2,
    #                             save_path='./dataset')
    # generate_train_test_dataset('./data/test/pelvis', padding=[592, 416], p='pelvis', t='test', interval=1,
    #                             save_path='./dataset')


if __name__ == '__main__':
    generate_dataset()
    check_dataset()

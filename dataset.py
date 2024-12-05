import os
import random

import SimpleITK as sitk
import numpy as np
import torch.utils.data
from tqdm import tqdm

dataser_path = './dataset'
MASK_POINT = []
MASK_AREA = []


def generate_mask(img_size, padding, val=False):
    mask = np.ones(img_size)
    # print('mask size: ', mask.shape)

    mask_size_w = random.randint(img_size[0] // 4, img_size[0] - 1)
    mask_size_h = random.randint(img_size[1] // 4, img_size[1] - 1)
    # mask_size = random.randint(img_size//16, img_size//4)

    if (mask_size_w == img_size[0] - 1 and mask_size_h == img_size[1] - 1) \
        or val == True:
        mask = np.expand_dims(1 - mask, axis=0)
        mask, location = mypadding(mask, x=padding[0], y=padding[1], v=1)
        return mask

    c_x = random.randint(0, img_size[0] - 1)
    c_y = random.randint(0, img_size[1] - 1)

    box_l_x = c_x - mask_size_w // 2
    box_l_y = c_y - mask_size_h // 2
    box_r_x = c_x + mask_size_w // 2
    box_r_y = c_y + mask_size_h // 2

    if box_l_x < 0:
        box_l_x = 0
    if box_l_y < 0:
        box_l_y = 0
    if box_r_x > img_size[0] - 1:
        box_r_x = img_size[0] - 1
    if box_r_y > img_size[1] - 1:
        box_r_y = img_size[1] - 1

    mask[box_l_y:box_r_y, box_l_x:box_r_x] = 0
    # print('*', c_y-mask_size//2, c_y + mask_size, c_x-mask_size//2,c_x + mask_size)
    mask = np.expand_dims(mask, axis=0)
    mask, location = mypadding(mask, x=padding[0], y=padding[1], v=1)

    MASK_POINT.append([c_x + location[0][0], c_y + location[0][1]])
    MASK_AREA.append([box_l_y, box_r_y, box_l_x, box_r_x])
    # print(mask.shape)
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
        # img = np.clip(img, 0, 1)
        return img
    elif type == 'cbct':
        # 后面试试直接 min max，不固定
        min_value = -1024
        max_value = 3000
        img = (img - min_value) / (max_value - min_value)
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
    brain_ds = os.listdir(path)

    size_heights = []
    size_widths = []

    cbct_min_values = []
    cbct_max_values = []

    ct_min_values = []
    ct_max_values = []

    for i in range(len(brain_ds)):
        bd = brain_ds[i]
        if bd == '.DS_Store':
            continue

        path_temp = os.path.join(path, bd)
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
    brain_ds = os.listdir(path)

    size_heights = []
    size_widths = []

    cbct_min_values = []
    cbct_max_values = []

    for i in range(len(brain_ds)):
        bd = brain_ds[i]
        if bd == '.DS_Store':
            continue

        path_temp = os.path.join(path, bd)
        path_cbct = os.path.join(path_temp, 'cbct.nii.gz')

        cbct = sitk.ReadImage(path_cbct)
        cbct = sitk.GetArrayFromImage(cbct)

        # print(cbct.shape)
        size_heights.append(cbct.shape[1])
        size_widths.append(cbct.shape[2])
        cbct_min_values.append(cbct.min())
        cbct_max_values.append(cbct.max())

    print('size')
    print('--height: {}-{}'.format(np.min(size_heights), np.max(size_heights)))
    print('--width: {}-{}'.format(np.min(size_widths), np.max(size_widths)))

    print('cbct')
    print('--value: {}-{}'.format(np.min(cbct_min_values), np.max(cbct_max_values)))


def window_transform(ct_array, windowWidth, windowCenter):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)

    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg


def process_nifti_image(file_path, normalize_type, padding):
    """
    读取、标准化并填充 NIfTI 图像。
    :param file_path: NIfTI 文件路径
    :param normalize_type: 标准化类型 ('cbct' 或 'ct')
    :param padding: 填充参数 (padding[0], padding[1])
    :return: 填充后的图像数组和位置信息
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    img_norm = normalize(img, type=normalize_type)
    img_padded, img_location = mypadding(img_norm, padding[0], padding[1])
    return img_padded, img_location


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

    cbct_vecs, ct_vecs, enhance_ct_vecs, mask_vecs, lct_vecs = [], [], [], [], []

    for bd in tqdm(dataset_dirs, total=len(dataset_dirs), desc="Processing cases"):
        case_path = os.path.join(path, bd)
        cbct_path = os.path.join(case_path, 'cbct.nii.gz')
        ct_path = os.path.join(case_path, 'ct.nii.gz')
        mask_path = os.path.join(case_path, 'mask.nii.gz')

        cbct_padded, img_location = process_nifti_image(cbct_path, 'cbct', padding)
        ct_padded, _ = process_nifti_image(ct_path, 'ct', padding)
        enhance_ct_padded, _ = process_nifti_image(ct_path, 'ct', padding)  # 使用 window_transform 后需要填充
        enhance_ct_padded = window_transform(enhance_ct_padded, 1000, 350)
        mask_padded, _ = process_nifti_image(mask_path, 'mask', padding)

        length = len(cbct_padded)

        for index in range(length):
            if index % interval != 0:
                continue

            cbct_2_5d = generate_2_5d_slices(cbct_padded, index, length)
            cbct_vecs.append(cbct_2_5d)
            lct_vecs.append(img_location)
            ct_vecs.append(ct_padded[index])
            enhance_ct_vecs.append(enhance_ct_padded[index])
            mask_vecs.append(mask_padded[index])

    # 转换为 NumPy 数组并调整维度
    cbct_np = np.array(cbct_vecs)
    ct_np = np.expand_dims(np.array(ct_vecs), axis=1)
    enhance_ct_np = np.expand_dims(np.array(enhance_ct_vecs), axis=1)
    mask_np = np.expand_dims(np.array(mask_vecs), axis=1).astype(np.float32)
    lct_np = np.concatenate(lct_vecs, axis=0).astype(np.float32)

    # 打印统计信息
    print(
        f"Shapes - cbct: {cbct_np.shape}, CT: {ct_np.shape}, Enhanced CT: {enhance_ct_np.shape}, Mask: {mask_np.shape},location: {lct_np.shape}")
    print(f"cbct min: {cbct_np.min()}, max: {cbct_np.max()}")
    print(f"ct min: {ct_np.min()}, max: {ct_np.max()}")
    print(f"enhance ct min: {enhance_ct_np.min()}, max: {enhance_ct_np.max()}")
    print(f"mask min: {mask_np.min()}, max: {mask_np.max()}")

    # 保存数据集
    dataset_name = os.path.join(save_path, f'synthRAD_interval_{interval}_{p}_{t}.npz')
    print(f"Saving dataset to: {dataset_name}")
    np.savez(dataset_name, img=np.concatenate((cbct_np, ct_np, enhance_ct_np, mask_np), axis=1), lct=lct_np)


def generate_ds():
    brain_shape = [304, 304]
    generate_train_test_dataset('./data/train/brain', padding=brain_shape, p='brain', t='train',
                                interval=2, save_path='./dataset')
    generate_train_test_dataset('./data/test/brain', padding=brain_shape, p='brain', t='test',
                                interval=1, save_path='./dataset')

    # generate_train_test_dataset('./data/train/pelvis', padding=[592, 416], p='pelvis', t='train', interval=2,
    #                             save_path='./dataset')
    # generate_train_test_dataset('./data/test/pelvis', padding=[592, 416], p='pelvis', t='test', interval=1,
    #                             save_path='./dataset')


def load_npz_data(dataset_path):
    """
    加载npz格式数据集
    :param dataset_path: 数据集路径
    :return: img 和 lct 数据
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ds = np.load(dataset_path)

    if 'img' not in ds or 'lct' not in ds:
        raise KeyError(f"The dataset at {dataset_path} must contain 'img' and 'lct' arrays.")

    return ds['img'], ds['lct']


def CreateDataset(dataset_path):
    print("dataset path:", dataset_path)
    img, lct = load_npz_data(dataset_path)
    print("img shape:", img.shape)
    print("lct shape:", lct.shape)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(img), torch.from_numpy(lct))
    return dataset


if __name__ == '__main__':
    generate_ds()
    # generate_valid_ds()

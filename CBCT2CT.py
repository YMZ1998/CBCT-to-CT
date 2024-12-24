import argparse
import os
import shutil
import time

import SimpleITK as sitk
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm


def post_process(out, location, mask, original_size, min_v=-1024, max_v=3000):
    out = np.squeeze(out)
    location = np.squeeze(location)
    mask = np.squeeze(mask) if mask is not None else None
    if original_size is not None:
        max_shape = max(original_size[0], original_size[1])
        if max_shape > out.shape[0] or max_shape > out.shape[1]:
            out = cv2.resize(out, [max_shape, max_shape], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, [max_shape, max_shape], interpolation=cv2.INTER_NEAREST)
            # sitk.WriteImage(sitk.GetImageFromArray(out), os.path.join(args.result_path, "out_resampled.nii.gz"))
            # sitk.WriteImage(sitk.GetImageFromArray(mask), os.path.join(args.result_path, "mask_resampled.nii.gz"))

    out = (out + 1) / 2
    out = out * (max_v - min_v) + min_v
    out = np.clip(out, min_v, max_v)

    y_min, x_min = int(location[1]), int(location[0])
    y_max, x_max = y_min + int(location[3]), x_min + int(location[2])

    out = out[y_min:y_max, x_min:x_max]
    mask = mask[y_min:y_max, x_min:x_max] if mask is not None else None

    return out, mask


def save_array_as_nii(array, file_path, reference=None):
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
    if reference is not None:
        sitk_image.CopyInformation(reference)
    sitk.WriteImage(sitk_image, file_path)


def generate_2_5d_slices(image, index, length):
    indices = [
        max(0, index - 2),
        max(0, index - 1),
        index,
        min(length - 1, index + 1),
        min(length - 1, index + 2)
    ]
    return np.array([image[i] for i in indices])


def img_normalize(img):
    min_value = np.min(img)
    max_value = np.max(img)
    img = (img - min_value) / (max_value - min_value)
    img = img * 2 - 1
    return img


def img_padding(img, x=288, y=288, v=0):
    h, w = img.shape[1], img.shape[2]

    padding_y = (y - h) // 2, (y - h) - (y - h) // 2
    padding_x = (x - w) // 2, (x - w) - (x - w) // 2

    padded_img = np.pad(img, ((0, 0), padding_y, padding_x), mode='constant', constant_values=v)

    img_location = np.array([padding_x[0], padding_y[0], w, h])
    img_location = np.expand_dims(img_location, 0)
    return padded_img, img_location


def load_data(cbct_path, mask_path, shape):
    # 读取CBCT图像
    origin_cbct = sitk.ReadImage(cbct_path)
    cbct_array = sitk.GetArrayFromImage(origin_cbct)
    original_size = origin_cbct.GetSize()
    print("Original size: ", original_size)
    original_spacing = origin_cbct.GetSpacing()
    print("Original spacing: ", original_spacing)

    origin_mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(origin_mask)
    mask_array[mask_array > 0] = 1

    # 如果CBCT尺寸大于目标尺寸，进行重采样
    if cbct_array.shape[1] > shape[0] or cbct_array.shape[2] > shape[1]:
        print("Resampling CBCT...")
        max_shape = max(cbct_array.shape[1], cbct_array.shape[2])
        # print("Max shape: ", max_shape)
        cbct_array, img_location = img_padding(cbct_array, max_shape, max_shape, -1)
        padding_cbct = sitk.GetImageFromArray(cbct_array)
        padding_cbct.SetSpacing(original_spacing)
        # sitk.WriteImage(padding_cbct, os.path.join(args.result_path, "padding_cbct.nii.gz"))

        mask_array, img_location = img_padding(mask_array, max_shape, max_shape)
        padding_mask = sitk.GetImageFromArray(mask_array)
        padding_mask.SetSpacing(original_spacing)
        # sitk.WriteImage(padding_mask, os.path.join(args.result_path, "padding_mask.nii.gz"))

        # 计算新的spacing保持物理尺寸一致
        new_spacing = [
            original_spacing[0] * cbct_array.shape[1] / shape[0],
            original_spacing[1] * cbct_array.shape[2] / shape[1],
            original_spacing[2]  # Z轴不变
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([shape[0], shape[1], cbct_array.shape[0]])  # 保留Z轴尺寸不变
        resampler.SetOutputSpacing(new_spacing)  # 使用计算出的新的spacing
        resampler.SetInterpolator(sitk.sitkLinear)
        cbct_resampled = resampler.Execute(padding_cbct)

        cbct_array = sitk.GetArrayFromImage(cbct_resampled)
        print("Resampled cbct shape: ", cbct_array.shape)
        print("New spacing: ", new_spacing)
        # sitk.WriteImage(cbct_resampled, os.path.join(args.result_path, "resample_cbct.nii.gz"))

        cbct_padded = img_normalize(cbct_array)

        print("Resample mask...")
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_resampled = resampler.Execute(padding_mask)
        mask_padded = sitk.GetArrayFromImage(mask_resampled)
        # sitk.WriteImage(mask_resampled, os.path.join(args.result_path, "resample_mask.nii.gz"))
        # print("Resampled mask shape: ", mask_padded.shape)
    else:
        cbct_array = img_normalize(cbct_array)
        cbct_padded, img_location = img_padding(cbct_array, shape[0], shape[1], -1)

        mask_padded, _ = img_padding(mask_array, shape[0], shape[1])

    return cbct_padded, img_location, mask_padded, origin_cbct


# def load_data(cbct_path, mask_path, shape):
#     cbct = sitk.ReadImage(cbct_path)
#     cbct = sitk.GetArrayFromImage(cbct)
#     print("cbct shape: ", cbct.shape)
#     cbct = img_normalize(cbct)
#     cbct_padded, img_location = img_padding(cbct, shape[0], shape[1], -1)
#
#     mask = sitk.ReadImage(mask_path)
#     mask = sitk.GetArrayFromImage(mask)
#     mask_padded, _ = img_padding(mask, shape[0], shape[1])
#     return cbct_padded, img_location, mask_padded


def val_onnx(args):
    if args.anatomy == 'brain':
        args.image_size = 320
    else:
        args.image_size = 480
    args.onnx_path = os.path.join(args.onnx_path, f'{args.anatomy}.onnx')
    shape = [args.image_size, args.image_size]

    os.makedirs(args.result_path, exist_ok=True)
    assert os.path.exists(args.cbct_path), f"CBCT file does not exist at {args.cbct_path}"
    assert os.path.exists(args.mask_path), f"Mask file does not exist at {args.mask_path}"
    assert os.path.exists(args.onnx_path), f"Onnx file does not exist at {args.onnx_path}"

    cbct_padded, img_location, mask_padded, origin_cbct = load_data(args.cbct_path, args.mask_path, shape)

    length = len(cbct_padded)
    cbct_vecs, mask_vecs, location_vecs = [], [], []
    for index in range(length):
        cbct_2_5d = generate_2_5d_slices(cbct_padded, index, length)
        cbct_vecs.append(cbct_2_5d)
        location_vecs.append(img_location)
        mask_vecs.append(mask_padded[index])
    cbct_batch = np.array(cbct_vecs[:]).astype(np.float32)
    mask_batch = np.expand_dims(np.array(mask_vecs[:]), axis=1).astype(np.float32)
    locations_batch = np.concatenate(location_vecs[:], axis=0).astype(np.float32)

    session = onnxruntime.InferenceSession(args.onnx_path)
    start_time = time.time()

    out_results = []
    for cbct, mask, image_locations in tqdm(zip(cbct_batch, mask_batch, locations_batch),
                                            total=len(cbct_batch)):
        cbct = np.expand_dims(cbct, 0)
        mask = np.expand_dims(mask, 0)
        output_name = session.get_outputs()[0].name
        ort_inputs = {session.get_inputs()[0].name: (cbct),
                      session.get_inputs()[1].name: (mask)}
        result = session.run([output_name], ort_inputs)[0].squeeze(0)

        out_cal, mask_cal = post_process(result, image_locations, mask, origin_cbct.GetSize())
        out_cal = np.where(mask_cal == 0, -1000, out_cal)

        out_results.append(np.expand_dims(out_cal, axis=0))

    out_results = np.concatenate(out_results, axis=0)

    if 'mhd' in args.cbct_path:
        predict_path = os.path.join(args.result_path, "predict.mhd")
        # mask_path = os.path.join(args.result_path, "origin_mask.mhd")
        # shutil.copy(args.mask_path, mask_path)
        # shutil.copy(args.mask_path.replace("mhd", "raw"), mask_path.replace("mhd", "raw"))
    else:
        predict_path = os.path.join(args.result_path, "predict.nii.gz")
        # mask_path = os.path.join(args.result_path, "origin_mask.nii.gz")
        # shutil.copy(args.mask_path, mask_path)

    save_array_as_nii(out_results, predict_path, origin_cbct)
    # if os.path.exists(args.mask_path.replace('mask', 'ct')):
    #     shutil.copy(args.mask_path.replace('mask', 'ct'), mask_path.replace('mask', 'ct'))
    total_time = time.time() - start_time
    print("time {}s".format(total_time))


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    # python CBCT2CT.py --cbct_path ./data/brain/test\2BA001\cbct.nii.gz --mask_path ./data/brain/test\2BA001\mask.nii.gz --result_path ./result --anatomy brain
    # python CBCT2CT.py --cbct_path ./data/pelvis/test\2PA001\cbct.nii.gz --mask_path ./data/pelvis/test\2PA001\mask.nii.gz --result_path ./result --anatomy pelvis
    # python CBCT2CT.py --cbct_path ./dist/test_data/cbct.nii.gz --mask_path ./dist/test_data/mask.nii.gz --result_path ./result --anatomy brain
    # CBCT2CT.exe  --cbct_path ./test_data/cbct.nii.gz --mask_path ./test_data/mask.nii.gz --result_path ./result --anatomy brain
    # CBCT2CT.exe --cbct_path ./test_data/brain/cbct.nii.gz --mask_path ./test_data/brain/mask.nii.gz --result_path ./result --anatomy brain
    # CBCT2CT.exe --cbct_path ./test_data/pelvis/cbct.nii.gz --mask_path ./test_data/pelvis/mask.nii.gz --result_path ./result --anatomy pelvis
    parser = argparse.ArgumentParser(
        prog='CBCT2CT.py',
        usage='%(prog)s [options] --cbct_path <path> --mask_path <path> --result_path <path>',
        description="CBCT generates pseudo CT.")
    # parser.add_argument("--image_size", default=320, type=int)
    parser.add_argument('--onnx_path', type=str, default='./checkpoint',
                        help="Path to onnx")
    parser.add_argument('--anatomy', choices=['brain', 'pelvis'], default='brain', help="The anatomy type")
    parser.add_argument('--cbct_path', type=str, required=True, help="Path to cbct file")
    parser.add_argument('--mask_path', type=str, required=True, help="Path to mask file")
    parser.add_argument('--result_path', type=str, required=True, help="Path to save results")
    args = parser.parse_args()
    print(args)
    val_onnx(args)

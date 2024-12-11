import argparse
import os
import shutil
import time

import SimpleITK as sitk
import numpy as np
import onnxruntime
from tqdm import tqdm


def to_numpy_squeeze(input_data):
    return np.squeeze(input_data)


def post_process(out, location, mask=None, min_v=-1024, max_v=3000):
    out = to_numpy_squeeze(out)
    location = to_numpy_squeeze(location)
    mask = to_numpy_squeeze(mask) if mask is not None else None

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


def img_normalize(img, type='cbct'):
    if type == 'cbct':
        min_value = np.min(img)
        max_value = np.max(img)
        img = (img - min_value) / (max_value - min_value)
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


def load_data(cbct_path, mask_path, shape):
    cbct = sitk.ReadImage(cbct_path)
    cbct = sitk.GetArrayFromImage(cbct)
    print(cbct.shape)
    cbct = img_normalize(cbct, type='cbct')
    cbct_padded, img_location = img_padding(cbct, shape[0], shape[1], -1)

    mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask)
    mask_padded, _ = img_padding(mask, shape[0], shape[1])
    return cbct_padded, img_location, mask_padded


def val_onnx(args):
    shape = [args.image_size, args.image_size]

    os.makedirs(args.result_path, exist_ok=True)
    assert os.path.exists(args.cbct_path), f"CBCT file does not exist at {args.cbct_path}"
    assert os.path.exists(args.mask_path), f"Mask file does not exist at {args.mask_path}"
    assert os.path.exists(args.onnx_path), f"Onnx file does not exist at {args.onnx_path}"

    cbct_padded, img_location, mask_padded = load_data(args.cbct_path,
                                                       args.mask_path,
                                                       shape)

    length = len(cbct_padded)
    cbct_vecs, mask_vecs, location_vecs = [], [], []
    for index in range(length):
        cbct_2_5d = generate_2_5d_slices(cbct_padded, index, length)
        cbct_vecs.append(cbct_2_5d)
        location_vecs.append(img_location)
        mask_vecs.append(mask_padded[index])
    cbct_batch = np.array(cbct_vecs[:])
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

        out_cal, mask_cal = post_process(result, image_locations, mask)
        out_cal = np.where(mask_cal == 0, -1000, out_cal)

        out_results.append(np.expand_dims(out_cal, axis=0))

    out_results = np.concatenate(out_results, axis=0)

    predict_path = os.path.join(args.result_path, "predict.nii.gz")
    mask_path = os.path.join(args.result_path, "origin_mask.nii.gz")

    save_array_as_nii(out_results, predict_path)
    shutil.copy(args.mask_path, mask_path)
    total_time = time.time() - start_time
    print("time {}s".format(total_time))


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    # data_path = './data/brain/test'
    # case_path = [os.path.join(data_path, p) for p in os.listdir(data_path)][0]
    # print(case_path)
    # cbct_path = os.path.join(case_path, 'cbct.nii.gz')
    # mask_path = os.path.join(case_path, 'mask.nii.gz')
    # result_path = "./result"
    # onnx_file_name = "./checkpoint/{}_best_model.onnx".format('efficientnet_b0')
    # # os.makedirs('./dist/checkpoint')
    # # shutil.copy(onnx_file_name, './dist/checkpoint/cbct2ct.onnx')
    # remove_and_create_dir(result_path)
    # print(f"python CBCT2CT.py --cbct_path {cbct_path} --mask_path {mask_path} --result_path {result_path}")
    # print(f"CBCT2CT.exe --cbct_path {cbct_path} --mask_path {mask_path} --result_path {result_path}")
    # python CBCT2CT.py --cbct_path ./data/brain/test\2BA001\cbct.nii.gz --mask_path ./data/brain/test\2BA001\mask.nii.gz --result_path ./result
    # CBCT2CT.exe --cbct_path ./test_data/cbct.nii.gz --mask_path ./test_data/mask.nii.gz --result_path ./result --onnx_path ./checkpoint/cbct2ct.onnx
    parser = argparse.ArgumentParser(
        prog='CBCT2CT.py',
        usage='%(prog)s [options] --cbct_path <path> --mask_path <path> --result_path <path>',
        description="CBCT generates pseudo CT.")
    parser.add_argument("--image_size", default=320, type=int)
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet_b0', help='efficientnet_b0')
    parser.add_argument('--onnx_path', type=str, default='./checkpoint/efficientnet_b0_best_model.onnx',
                        help="Path to onnx")
    parser.add_argument('--cbct_path', type=str, required=True, help="Path to cbct file")
    parser.add_argument('--mask_path', type=str, required=True, help="Path to mask file")
    parser.add_argument('--result_path', type=str, required=True, help="Path to save results")
    args = parser.parse_args()
    print(args)
    val_onnx(args)

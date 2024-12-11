import argparse
import os
import shutil
import time

import SimpleITK as sitk
import numpy as np
import onnxruntime
from tqdm import tqdm

from dataset import generate_2_5d_slices, img_normalize, mypadding
from image_metrics import ImageMetrics, compute_metrics
from model_tester import save_array_as_nii
from parse_args import remove_and_create_dir
from post_process import post_process
from predict import load_data


def load_data(cbct_path, ct_path, mask_path, shape):
    cbct = sitk.ReadImage(cbct_path)
    cbct = sitk.GetArrayFromImage(cbct)
    print(cbct.shape)
    cbct = img_normalize(cbct, type='cbct')
    cbct_padded, img_location = mypadding(cbct, shape[0], shape[1], -1)
    if args.ct_path is not None:
        ct = sitk.ReadImage(ct_path)
        ct = sitk.GetArrayFromImage(ct)
        ct_padded, _ = mypadding(ct, shape[0], shape[1], -1000)
        ct_padded = img_normalize(ct_padded, type='ct')
    else:
        ct_padded = np.zeros_like(cbct_padded)

    mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask)
    mask_padded, _ = mypadding(mask, shape[0], shape[1])
    return cbct_padded, img_location, ct_padded, mask_padded


def val_onnx(args):
    shape = [args.image_size, args.image_size]

    os.makedirs(args.result_path, exist_ok=True)
    assert os.path.exists(args.cbct_path), f"CBCT file does not exist at {args.cbct_path}"
    if args.ct_path is not None:
        assert os.path.exists(args.ct_path), f"CT file does not exist at {args.ct_path}"
    assert os.path.exists(args.mask_path), f"Mask file does not exist at {args.mask_path}"

    cbct_padded, img_location, ct_padded, mask_padded = load_data(args.cbct_path, args.ct_path,
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

    # onnx_file_name = "./checkpoint/{}_best_model.onnx".format(args.arch)
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

        out_cal, ct_cal, mask_cal = post_process(result, None, image_locations, mask)
        out_cal = np.where(mask_cal == 0, -1000, out_cal)

        out_results.append(np.expand_dims(out_cal, axis=0))

    out_results = np.concatenate(out_results, axis=0)

    origin_ct_path = os.path.join(args.result_path, "origin_ct.nii.gz")
    predict_path = os.path.join(args.result_path, "predict.nii.gz")
    mask_path = os.path.join(args.result_path, "origin_mask.nii.gz")

    save_array_as_nii(out_results, predict_path)
    shutil.copy(args.mask_path, mask_path)
    if args.ct_path is not None:
        shutil.copy(args.ct_path, origin_ct_path)
        compute_metrics(origin_ct_path, predict_path, mask_path)
    total_time = time.time() - start_time
    print("time {}s".format(total_time))


if __name__ == '__main__':
    data_path = './data/brain/test'
    case_path = [os.path.join(data_path, p) for p in os.listdir(data_path)][0]
    print(case_path)
    cbct_path = os.path.join(case_path, 'cbct.nii.gz')
    ct_path = os.path.join(case_path, 'ct.nii.gz')
    mask_path = os.path.join(case_path, 'mask.nii.gz')
    result_path = "./result"
    remove_and_create_dir(result_path)
    print(
        f"python val_onnx.py --cbct_path {cbct_path} --ct_path {ct_path} --mask_path {mask_path} --result_path {result_path}")
    print(f"python val_onnx.py --cbct_path {cbct_path} --mask_path {mask_path} --result_path {result_path}")

    parser = argparse.ArgumentParser(
        prog='val_onnx.py',
        usage='%(prog)s [options] --cbct_path <path> --mask_path <path> --result_path <path>',
        description="CBCT generates pseudo CT.")
    parser.add_argument("--image_size", default=320, type=int)
    parser.add_argument('--onnx_path', type=str, default='"./checkpoint/efficientnet_b0_best_model.onnx"',
                        help="Path to onnx")
    parser.add_argument('--cbct_path', type=str, required=True, help="Path to cbct file")
    parser.add_argument('--ct_path', type=str, help="Path to CT file")
    parser.add_argument('--mask_path', type=str, required=True, help="Path to mask file")
    parser.add_argument('--result_path', type=str, required=True, help="Path to save results")
    args = parser.parse_args()
    print(args)
    val_onnx(args)

import os

from nii2dcm.run import run_nii2dcm

nii_file = r"D:\Python_code\CBCT-to-CT\dist\test_data\brain\cbct.nii.gz"
out_path = './output_dicom2'
os.makedirs(out_path, exist_ok=True)

run_nii2dcm(nii_file, out_path, dicom_type='SVR')
import shutil

# nii2dcm D:\Python_code\CBCT-to-CT\dist\test_data\pelvis\cbct.nii.gz output_dicom/ --d MR
# nii2dcm nifti-file.nii.gz dicom-output-directory/ --dicom-type MR

# import nibabel as nib
# import pydicom
# import numpy as np
# import os
# import uuid
#
#
# def remove_and_create_dir(path):
#     if os.path.exists(path):
#         shutil.rmtree(path)
#     os.makedirs(path, exist_ok=True)
#
#
# # 读取NIfTI文件
# nii_file = r"D:\Python_code\CBCT-to-CT\dist\test_data\pelvis\cbct.nii.gz"
# img = nib.load(nii_file)
# data = img.get_fdata()  # 获取数据，形状为 (X, Y, Z)
#
# # 获取数据的维度
# x_dim, y_dim, z_dim = data.shape
#
# # 创建保存DICOM文件的文件夹
# output_dir = 'output_dicom2'
#
# remove_and_create_dir(output_dir)
#
# # 循环遍历每个切片，并为每个切片创建DICOM文件
# for i in range(z_dim):
#     slice_2d = data[:, :, i]  # 获取第i层的二维切片
#
#     # 确保数据为16位整型（假设数据值已归一化为16位）
#     slice_2d = np.array(slice_2d, dtype=np.uint16)
#
#     # 创建DICOM对象
#     dicom_file = pydicom.Dataset()
#
#     # 设置DICOM的基本信息（示例中可以添加更多字段，具体取决于你的数据需求）
#     dicom_file.PatientName = "Test Patient"
#     dicom_file.PatientID = "123456"
#     dicom_file.StudyDate = "20250103"
#     dicom_file.Modality = "MR"  # 设置为MRI，若为CT则为"CT"
#     dicom_file.Rows, dicom_file.Columns = slice_2d.shape
#     dicom_file.SamplesPerPixel = 1  # 单通道图像
#     dicom_file.PhotometricInterpretation = "MONOCHROME2"
#     dicom_file.BitsAllocated = 16
#     dicom_file.BitsStored = 16
#     dicom_file.HighBit = 15
#
#     # 设置SOPClassUID和SOPInstanceUID
#     # dicom_file.SOPClassUID = pydicom.uid.ImplicitVRBigEndian
#     # dicom_file.SOPInstanceUID = pydicom.uid.generate_uid()
#
#     # 设置像素数据
#     dicom_file.PixelData = slice_2d.tobytes()
#
#     # 必须显式设置这两个属性以避免保存错误
#     dicom_file.is_little_endian = True  # 设置为小端字节序
#     dicom_file.is_implicit_VR = False  # 使用隐式VR
#
#     # 保存DICOM文件
#     dicom_file.save_as(os.path.join(output_dir, f"slice_{i + 1:03d}.dcm"))
#
# print(f"所有DICOM切片已保存到 {output_dir}")

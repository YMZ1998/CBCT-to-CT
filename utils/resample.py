import SimpleITK as sitk
import numpy as np


def resample(image, mask, shape):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    print("Original size: ", original_size)
    print("Original spacing: ", original_spacing)
    print("Original origin: ", image.GetOrigin())

    new_spacing = [
        original_spacing[0] * original_size[0] / shape[0],
        original_spacing[1] * original_size[1] / shape[1],
        original_spacing[2]
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([shape[0], shape[1], original_size[2]])
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)

    resampled_image = resampler.Execute(image)

    print("Resampled size: ", resampled_image.GetSize())
    print("Resampled spacing: ", resampled_image.GetSpacing())
    print("Resampled origin: ", resampled_image.GetOrigin())

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled_mask = resampler.Execute(mask)

    return resampled_image, resampled_mask


if __name__ == '__main__':
    image_path = r"./data/brain.nii.gz"
    mask_path = r"./data/brain_mask.nii.gz"
    # image_path = r"./data/pelvis.nii.gz"
    # mask_path = r"./data/pelvis_mask.nii.gz"

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    resampled_image, resampled_mask = resample(image, mask, [480, 480])

    resampled_image = sitk.GetArrayFromImage(resampled_image)
    resampled_image = resampled_image - np.min(resampled_image)
    print(np.min(resampled_image))
    resampled_image = sitk.GetImageFromArray(resampled_image)

    sitk.WriteImage(resampled_image, "./data/case/cbct.nii.gz")
    sitk.WriteImage(resampled_mask, "./data/case/mask.nii.gz")

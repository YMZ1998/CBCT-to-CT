import numpy as np
from skimage import morphology
from skimage.measure import label


def normalize(img, min_, max_):
    return (img - min_) / (max_ - min_)


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def get_3d_mask(img, th=500, width=2):
    mask = np.zeros(img.shape).astype(int)
    mask[img > th] = 1

    mask = morphology.binary_opening(mask, )

    remove_holes = morphology.remove_small_holes(mask, area_threshold=width ** 3)

    largest_cc = getLargestCC(remove_holes)

    return img, largest_cc.astype(int)


if __name__ == '__main__':
    import SimpleITK as sitk

    ct = sitk.ReadImage('./dist/test_data/brain/cbct.nii.gz')
    ct_array = sitk.GetArrayFromImage(ct)
    ct_array, mask = get_3d_mask(ct_array)
    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(ct)
    sitk.WriteImage(mask, './dist/test_data/brain/cbct_mask.nii.gz')

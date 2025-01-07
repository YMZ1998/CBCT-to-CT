import numpy as np
from skimage import morphology
from skimage.measure import label
from skimage.filters import threshold_otsu


def normalize(img, min_, max_):
    return (img - min_) / (max_ - min_)


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def get_3d_mask(img):
    mask = np.zeros(img.shape).astype(int)
    otsu_threshold = threshold_otsu(img)
    # print(otsu_threshold,np.min(img) + 300)
    mask[img > otsu_threshold] = 1

    mask = morphology.binary_opening(mask, )

    remove_holes = morphology.remove_small_holes(mask, area_threshold=4)

    largest_cc = getLargestCC(remove_holes)

    return img, largest_cc.astype(int)


if __name__ == '__main__':
    import SimpleITK as sitk

    case = 'brain'
    # case='pelvis'

    ct = sitk.ReadImage(f'./data/{case}.nii.gz')
    ct_array = sitk.GetArrayFromImage(ct)
    ct_array, mask = get_3d_mask(ct_array)
    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(ct)
    sitk.WriteImage(mask, f'./data/{case}_mask.nii.gz')

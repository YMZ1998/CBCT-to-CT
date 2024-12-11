from typing import Optional

import SimpleITK as sitk
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop


class ImageMetrics:
    def __init__(self, dynamic_range=[-1024., 3000.]):
        # Use fixed wide dynamic range
        self.dynamic_range = dynamic_range

    def score_patient(self, gt_array, pred_array, mask_array):

        # Calculate image metrics
        mae_value = self.mae(gt_array,
                             pred_array,
                             mask_array)

        psnr_value = self.psnr(gt_array,
                               pred_array,
                               mask_array,
                               use_population_range=True)

        ssim_value = self.ssim(gt_array,
                               pred_array,
                               mask_array)
        return {
            'mae': mae_value,
            'ssim': ssim_value,
            'psnr': psnr_value
        }

    def mae(self,
            gt: np.ndarray,
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mean Absolute Error (MAE)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).

        Returns
        -------
        mae : float
            mean absolute error.

        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

        mae_value = np.sum(np.abs(gt * mask - pred * mask)) / mask.sum()
        return float(mae_value)

    def psnr(self,
             gt: np.ndarray,
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        """
        Compute Peak Signal to Noise Ratio metric (PSNR)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
        use_population_range : bool, optional
            When a predefined population wide dynamic range should be used.
            gt and pred will also be clipped to these values.

        Returns
        -------
        psnr : float
            Peak signal to noise ratio..

        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

        if use_population_range:
            dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]

            # Clip gt and pred to the dynamic range
            gt = np.where(gt < self.dynamic_range[0], self.dynamic_range[0], gt)
            gt = np.where(gt > self.dynamic_range[1], self.dynamic_range[1], gt)
            pred = np.where(pred < self.dynamic_range[0], self.dynamic_range[0], pred)
            pred = np.where(pred > self.dynamic_range[1], self.dynamic_range[1], pred)
        else:
            dynamic_range = gt.max() - gt.min()

        # apply mask
        gt = gt[mask == 1]
        pred = pred[mask == 1]
        psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
        return float(psnr_value)

    def ssim(self,
             gt: np.ndarray,
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Structural Similarity Index Metric (SSIM)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).

        Returns
        -------
        ssim : float
            structural similarity index metric.

        """
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

            # Mask gt and pred
            gt = np.where(mask == 0, min(self.dynamic_range), gt)
            pred = np.where(mask == 0, min(self.dynamic_range), pred)

        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        ssim_value_full, ssim_map = structural_similarity(gt, pred, data_range=dynamic_range, full=True)

        if mask is not None:
            # Follow skimage implementation of calculating the mean value:
            # crop(ssim_map, pad).mean(dtype=np.float64), with pad=3 by default.
            pad = 3
            ssim_value_masked = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)
            return float(ssim_value_masked)
        else:
            return float(ssim_value_full)


def compute_metrics(origin_ct_path, predict_path, mask_path):
    metrics = ImageMetrics()
    gt = sitk.ReadImage(origin_ct_path)
    pred = sitk.ReadImage(predict_path)
    mask = sitk.ReadImage(mask_path)
    gt_array = sitk.GetArrayFromImage(gt)
    pred_array = sitk.GetArrayFromImage(pred)
    mask_array = sitk.GetArrayFromImage(mask)
    print(metrics.score_patient(gt_array, pred_array, mask_array))


if __name__ == '__main__':
    origin_ct_path = "./result/origin_ct.nii.gz"
    predict_path = "./result/predict.nii.gz"
    mask_path = "./result/origin_mask.nii.gz"
    compute_metrics(origin_ct_path, predict_path, mask_path)

import sys

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from image_metrics import ImageMetrics
from post_process import post_process

plt.switch_backend('agg')

synthrad_metrics_stage1 = ImageMetrics(dynamic_range=[-150., 850.])
synthrad_metrics_stage2_stage3 = ImageMetrics(dynamic_range=[-1024., 3000.])


def save_array_as_nii(array, file_path, reference=None):
    """
    Save a NumPy array as a NIfTI file using SimpleITK.

    Args:
        array (numpy.ndarray): The array to save.
        file_path (str): The path to save the NIfTI file.
        reference (SimpleITK.Image): Optional reference image for setting metadata.
    """
    sitk_image = sitk.GetImageFromArray(array)
    if reference is not None:
        sitk_image.CopyInformation(reference)
    sitk.WriteImage(sitk_image, file_path)


class ModelTester:
    def __init__(self, stage1, stage2, resbranch, device, epoch_stage1, epoch_stage2, logger, save_all=False):
        self.stage1 = stage1
        self.stage2 = stage2
        self.resbranch = resbranch
        self.device = device
        self.epoch_stage1 = epoch_stage1
        self.epoch_stage2 = epoch_stage2
        self.logger = logger
        self.save_all = save_all

    def load_model_weights(self, weight_path):
        print(f"Loading checkpoint from {weight_path}")
        checkpoint = torch.load(weight_path, weights_only=False, map_location='cpu')
        self.stage1.load_state_dict(checkpoint['model_stage1'])
        self.stage2.load_state_dict(checkpoint['model_stage2'])
        self.resbranch.load_state_dict(checkpoint['model_resbranch'])
    def save_visualization(self, epoch, iteration, metrics, show_cbct, show_origin_ct, show_stage1_out,
                           mask, show_enhanced_ct, show_stage2_out=None, show_stage3_out=None, show_final_out=None):
        # print("show_cbct", np.min(show_cbct), np.max(show_cbct))
        # print("show_origin_ct", np.min(show_origin_ct), np.max(show_origin_ct))
        # print("show_stage1_out", np.min(show_stage1_out), np.max(show_stage1_out))
        # print("show_stage2_out", np.min(show_stage2_out), np.max(show_stage2_out))
        # print("show_stage3_out", np.min(show_stage3_out), np.max(show_stage3_out))
        # print("mask", np.min(mask), np.max(mask))
        fig = plt.figure(figsize=(9, 6), dpi=100, tight_layout=True)
        fig.suptitle(f'epoch {epoch} psnr: {metrics["psnr"]:.4f} ssim: {metrics["ssim"]:.4f} mae:{metrics["mae"]:.4f}')

        # 显示 CBCT 图像
        plt.subplot(2, 4, 1)
        plt.axis("off")
        plt.imshow(show_cbct[1], cmap="gray")
        plt.title("CBCT")

        # 显示原始 CT 图像
        plt.subplot(2, 4, 2)
        plt.axis("off")
        plt.imshow(show_origin_ct, cmap="gray")
        plt.title("Original CT")

        # 显示增强后的 CT 图像
        plt.subplot(2, 4, 3)
        plt.axis("off")
        plt.imshow(show_enhanced_ct, cmap="gray")
        plt.title("Enhanced CT")

        # 显示 mask
        plt.subplot(2, 4, 4)
        plt.axis("off")
        if mask is not None:
            plt.imshow(mask, cmap="gray")
        plt.title("Mask")

        # 显示阶段1的输出图像
        plt.subplot(2, 4, 5)
        plt.axis("off")
        if mask is not None:
            plt.imshow(np.where(mask == 0, -1, show_stage1_out), cmap="gray")
        else:
            plt.imshow(show_stage1_out, cmap="gray")
        plt.title("Stage 1 Output")

        # 显示阶段2的输出图像
        if show_stage2_out is not None:
            plt.subplot(2, 4, 6)
            plt.axis("off")
            if mask is not None:
                plt.imshow(np.where(mask == 0, -1, show_stage2_out), cmap="gray")
            else:
                plt.imshow(show_stage2_out, cmap="gray")
            plt.title("Stage 2 Output")

        # 显示阶段3的输出图像
        if show_stage3_out is not None:
            plt.subplot(2, 4, 7)
            plt.axis("off")
            if mask is not None:
                plt.imshow(np.where(mask == 0, -1, show_stage3_out), cmap="gray")
            else:
                plt.imshow(show_stage3_out, cmap="gray")
            plt.title("Stage 3 Output")

        # 显示最终的输出图像
        if show_final_out is not None:
            plt.subplot(2, 4, 8)
            plt.axis("off")
            if mask is not None:
                plt.imshow(np.where(mask == 0, -1, show_final_out), cmap="gray")
            else:
                plt.imshow(show_final_out, cmap="gray")
            plt.title("Final Output")
        # plt.show()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f"visualization/epoch{epoch}_iteration{iteration}.png", dpi=100)
        plt.clf()
        plt.close(fig)

    def process_output(self, stage1_out, stage2_out, stage3_out, final_out, origin_cbct, origin_mask, origin_ct,
                       enhance_ct,
                       origin_location, epoch, iteration, slice_index, stage, metrics):
        """处理每一阶段的输出、计算指标并保存可视化结果"""
        if stage == 1:
            out_cal, ct_cal, mask_cal = post_process(stage1_out, enhance_ct, origin_location, origin_mask, min_v=-150,
                                                     max_v=850)
            global_metrics = synthrad_metrics_stage1.score_patient(ct_cal, out_cal, mask_cal)
        elif stage == 2:
            out_cal, ct_cal, mask_cal = post_process(stage2_out, origin_ct, origin_location, origin_mask)
            global_metrics = synthrad_metrics_stage2_stage3.score_patient(ct_cal, out_cal, mask_cal)
        else:
            out_cal, ct_cal, mask_cal = post_process(final_out, origin_ct, origin_location, origin_mask)
            global_metrics = synthrad_metrics_stage2_stage3.score_patient(ct_cal, out_cal, mask_cal)

        if iteration == slice_index or (self.save_all and iteration % 10 == 0):
            # 处理和转换图像数据
            show_cbct = origin_cbct.detach().cpu().squeeze().numpy()
            show_origin_ct = origin_ct.detach().cpu().squeeze().numpy()
            mask = origin_mask.detach().cpu().squeeze().numpy()
            show_stage1_out = stage1_out.detach().cpu().squeeze().numpy()
            show_enhanced_ct = enhance_ct.detach().cpu().squeeze().numpy()
            show_stage2_out = stage2_out.detach().cpu().squeeze().numpy() if stage2_out is not None else None
            show_stage3_out = stage3_out.detach().cpu().squeeze().numpy() if stage3_out is not None else None
            show_final_out_out = final_out.detach().cpu().squeeze().numpy() if final_out is not None else None

            # 保存图像
            self.save_visualization(epoch, iteration, global_metrics, show_cbct, show_origin_ct,
                                    show_stage1_out, mask, show_enhanced_ct, show_stage2_out, show_stage3_out,
                                    show_final_out_out)

        metrics[0, 0, iteration] = global_metrics['psnr']
        metrics[0, 1, iteration] = global_metrics['ssim']
        metrics[0, 2, iteration] = global_metrics['mae']

        return metrics

    def test(self, data_loader_test, epoch):
        self.stage1.eval()
        self.stage2.eval()
        self.resbranch.eval()

        ds_len = len(data_loader_test)
        metrics = np.zeros((1, 3, ds_len))
        slice_index = 120

        with torch.no_grad():
            for iteration, (images, image_locations) in enumerate(tqdm(data_loader_test, file=sys.stdout)):

                images = images.to(self.device)
                origin_cbct, origin_ct, enhance_ct, mask = torch.split(images, [3, 1, 1, 1], dim=1)
                # mask.fill_(1.0)
                stage1_out = self.stage1(origin_cbct * mask)

                if epoch <= self.epoch_stage1:
                    metrics = self.process_output(stage1_out, None, None, None, origin_cbct, mask, origin_ct,
                                                  enhance_ct,
                                                  image_locations, epoch, iteration, slice_index, stage=1,
                                                  metrics=metrics)
                elif epoch <= self.epoch_stage2:
                    stage2_out = self.stage2(stage1_out * mask)
                    metrics = self.process_output(stage1_out, stage2_out, None, None, origin_cbct, mask, origin_ct,
                                                  enhance_ct, image_locations, epoch, iteration, slice_index, stage=2,
                                                  metrics=metrics)
                else:
                    stage2_out = self.stage2(stage1_out * mask)
                    stage3_out = self.resbranch(origin_cbct * mask)
                    # stage3_out = (stage2_out + stage3_out) / 2
                    final_out = torch.tanh(stage2_out + stage3_out)
                    metrics = self.process_output(stage1_out, stage2_out, stage3_out, final_out, origin_cbct, mask,
                                                  origin_ct,
                                                  enhance_ct, image_locations, epoch, iteration, slice_index, stage=3,
                                                  metrics=metrics)

            # 打印和记录最终测试结果
            avg_psnr = np.nanmean(metrics[0][0])
            avg_ssim = np.nanmean(metrics[0][1])
            avg_mae = np.nanmean(metrics[0][2])

            print(f'psnr: {avg_psnr} ssim: {avg_ssim} mae: {avg_mae}')
            log_str = f"test epoch:{epoch} psnr {avg_psnr} ssim:{avg_ssim} mae:{avg_mae}"
            self.logger.info(log_str)

        return {'psnr': avg_psnr, 'ssim': avg_ssim, 'mae': avg_mae}

    def predict(self, data_loader_test):
        self.stage1.eval()
        self.stage2.eval()
        self.resbranch.eval()

        out_results, ct_results, mask_results = [], [], []
        with torch.no_grad():
            for iteration, (images, image_locations) in enumerate(tqdm(data_loader_test, file=sys.stdout)):
                images = images.to(self.device)
                origin_cbct, origin_ct, enhance_ct, mask = torch.split(images, [3, 1, 1, 1], dim=1)
                # mask.fill_(1)

                stage1_out = self.stage1(origin_cbct * mask)

                stage2_out = self.stage2(stage1_out * mask)
                stage3_out = self.resbranch(origin_cbct * mask)
                final_out = torch.tanh(stage2_out + stage3_out)

                out_cal, ct_cal, mask_cal = post_process(final_out, origin_ct, image_locations, mask)

                out_cal = np.where(mask_cal == 0, -1000, out_cal)

                out_results.append(np.expand_dims(out_cal, axis=0))
                ct_results.append(np.expand_dims(ct_cal, axis=0))
                mask_results.append(np.expand_dims(mask_cal, axis=0))

        out_results = np.concatenate(out_results, axis=0)
        ct_results = np.concatenate(ct_results, axis=0)
        mask_results = np.concatenate(mask_results, axis=0)
        print(out_results.shape)
        save_array_as_nii(out_results, "./result/predict.nii.gz")
        save_array_as_nii(ct_results, "./result/origin_ct.nii.gz")
        save_array_as_nii(mask_results, "./result/origin_mask.nii.gz")


if __name__ == '__main__':
    from predict import predict
    from test import test

    # test()
    predict()

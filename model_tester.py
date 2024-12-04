import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from post_process import process
from image_metrics import *

synthrad_metrics_stage1 = ImageMetrics(dynamic_range=[-150., 850.])
synthrad_metrics_stage2 = ImageMetrics(dynamic_range=[-1024., 3000.])


class ModelTester:
    def __init__(self, stage1, stage2, resbranch, device, epoch_stage1, epoch_stage2, logger):
        self.stage1 = stage1
        self.stage2 = stage2
        self.resbranch = resbranch
        self.device = device
        self.epoch_stage1 = epoch_stage1
        self.epoch_stage2 = epoch_stage2
        self.logger = logger

    def save_visualization(self, epoch, iteration, slice_index, metrics, show_mr, show_ct, show_out1, show_mask,
                           show_ct2=None, show_out2=None):
        """保存视觉化结果"""
        fig = plt.figure(figsize=(15, 3), dpi=100, tight_layout=True)
        fig.suptitle(f'epoch {epoch} psnr: {metrics["psnr"]:.4f} ssim: {metrics["ssim"]:.4f} mae:{metrics["mae"]:.4f}')

        plt.subplot(1, 5, 1)
        plt.axis("off")
        plt.imshow(show_mr[2], cmap="gray")

        plt.subplot(1, 5, 2)
        plt.axis("off")
        plt.imshow(show_ct, cmap="gray")

        plt.subplot(1, 5, 3)
        plt.axis("off")
        plt.imshow(show_out1 * show_mask, cmap="gray")

        if show_ct2 is not None:
            plt.subplot(1, 5, 4)
            plt.axis("off")
            plt.imshow(show_ct2, cmap="gray")

        if show_out2 is not None:
            plt.subplot(1, 5, 5)
            plt.axis("off")
            plt.imshow(show_out2 * show_mask, cmap="gray")

        plt.subplots_adjust(top=0.85)
        plt.savefig(f"visualization/epoch{epoch}_iteration{iteration}.png", dpi=300)
        plt.clf()
        plt.close(fig)

    def process_output(self, out_global, out_second, samp_mr_gpu, mask_gpu, samp_ct_gpu, samp_ct2_gpu, samp_lct, epoch,
                       iteration,
                       slice_index, stage, metrics):
        """处理每一阶段的输出、计算指标并保存可视化结果"""
        if stage == 1:
            out_cal, ct_cal, mask_cal = process(out_global, samp_ct2_gpu, samp_lct, mask_gpu, min_v=-150, max_v=850)
            m_global = synthrad_metrics_stage1.score_patient(ct_cal, out_cal, mask_cal)
        else:
            out_cal, ct_cal, mask_cal = process(out_second, samp_ct_gpu, samp_lct, mask_gpu)
            m_global = synthrad_metrics_stage2.score_patient(ct_cal, out_cal, mask_cal)

        if iteration == slice_index:
            show_mr = samp_mr_gpu.detach().cpu().squeeze().numpy()
            show_ct = samp_ct_gpu.detach().cpu().squeeze().numpy()
            show_mask = mask_gpu.detach().cpu().squeeze().numpy()
            show_out1 = out_global.detach().cpu().squeeze().numpy()
            show_ct2 = samp_ct2_gpu.detach().cpu().squeeze().numpy()
            show_out2 = out_second.detach().cpu().squeeze().numpy() if out_second is not None else None

            # 保存图像
            self.save_visualization(epoch, iteration, slice_index, m_global, show_mr, show_ct, show_out1, show_mask,
                                    show_ct2, show_out2)

        metrics[0, 0, iteration] = m_global['psnr']
        metrics[0, 1, iteration] = m_global['ssim']
        metrics[0, 2, iteration] = m_global['mae']

        return metrics

    def test(self, data_loader_test, epoch):
        """封装后的测试函数"""
        self.stage1.eval()
        self.stage2.eval()
        self.resbranch.eval()

        ds_len = len(data_loader_test)
        metrics = np.zeros((1, 3, ds_len))
        slice_index = 120

        with torch.no_grad():
            for iteration, (samp_img, samp_lct) in enumerate(tqdm(data_loader_test)):

                samp_img_gpu = samp_img.to(self.device)
                samp_mr_gpu, samp_ct_gpu, samp_ct2_gpu, mask_gpu = torch.split(samp_img_gpu, [5, 1, 1, 1], dim=1)

                out_global = self.stage1(samp_mr_gpu * mask_gpu)

                if epoch <= self.epoch_stage1:
                    metrics = self.process_output(out_global, None, samp_mr_gpu, mask_gpu, samp_ct_gpu, samp_ct2_gpu,
                                                  samp_lct,
                                                  epoch, iteration, slice_index, stage=1, metrics=metrics)
                elif epoch <= self.epoch_stage2:
                    out_second = self.stage2(out_global * mask_gpu)
                    metrics = self.process_output(out_global, out_second, samp_mr_gpu, mask_gpu, samp_ct_gpu,
                                                  samp_ct2_gpu, samp_lct,
                                                  epoch, iteration, slice_index, stage=2, metrics=metrics)
                else:
                    out_second = self.stage2(out_global * mask_gpu)
                    out_third = self.resbranch(samp_mr_gpu * mask_gpu)
                    out = out_second + out_third
                    metrics = self.process_output(out, out_second, samp_mr_gpu, mask_gpu, samp_ct_gpu, samp_ct2_gpu,
                                                  samp_lct, epoch,
                                                  iteration, slice_index, stage=3, metrics=metrics)

            # 打印和记录最终测试结果
            avg_psnr = np.nanmean(metrics[0][0])
            avg_ssim = np.nanmean(metrics[0][1])
            avg_mae = np.nanmean(metrics[0][2])

            print(f'psnr: {avg_psnr} ssim: {avg_ssim} mae: {avg_mae}')
            log_str = f"test epoch:{epoch} psnr {avg_psnr} ssim:{avg_ssim} mae:{avg_mae}"
            self.logger.info(log_str)

        return {'psnr': avg_psnr, 'ssim': avg_ssim, 'mae': avg_mae}

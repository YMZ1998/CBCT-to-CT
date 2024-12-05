import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from image_metrics import *
from post_process import process

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

    def save_visualization(self, epoch, iteration, slice_index, metrics, show_cbct, show_origin_ct, show_stage1_out,
                           mask, show_enhanced_ct, show_stage2_out=None):
        # 创建一个图形
        fig = plt.figure(figsize=(15, 3), dpi=100, tight_layout=True)
        fig.suptitle(f'epoch {epoch} psnr: {metrics["psnr"]:.4f} ssim: {metrics["ssim"]:.4f} mae:{metrics["mae"]:.4f}')

        # 显示 CBCT 图像
        plt.subplot(1, 5, 1)
        plt.axis("off")
        plt.imshow(show_cbct[2], cmap="gray")
        plt.title("CBCT Slice")

        # 显示原始 CT 图像
        plt.subplot(1, 5, 2)
        plt.axis("off")
        plt.imshow(show_origin_ct, cmap="gray")
        plt.title("Original CT")

        # 显示增强后的 CT 图像
        plt.subplot(1, 5, 3)
        plt.axis("off")
        plt.imshow(show_enhanced_ct, cmap="gray")
        plt.title("Enhanced CT")

        # 显示阶段1的输出图像
        plt.subplot(1, 5, 4)
        plt.axis("off")
        plt.imshow(show_stage1_out * mask, cmap="gray")
        plt.title("Stage 1 Output")

        # 如果有阶段2的输出，显示阶段2的图像
        if show_stage2_out is not None:
            plt.subplot(1, 5, 5)
            plt.axis("off")
            plt.imshow(show_stage2_out * mask, cmap="gray")
            plt.title("Stage 2 Output")

        # 调整图形布局，防止标题被遮挡
        plt.subplots_adjust(top=0.85)

        # 保存图像
        plt.savefig(f"visualization/epoch{epoch}_iteration{iteration}.png", dpi=300)

        # 清空图形缓存并关闭图形
        plt.clf()
        plt.close(fig)

    def process_output(self, stage1_out, stage2_out, origin_cbct, origin_mask, origin_ct, enhance_ct, origin_location,
                       epoch, iteration, slice_index, stage, metrics, save_all=False):
        """处理每一阶段的输出、计算指标并保存可视化结果"""
        if stage == 1:
            out_cal, ct_cal, mask_cal = process(stage1_out, enhance_ct, origin_location, origin_mask, min_v=-150,
                                                max_v=850)
            global_metrics = synthrad_metrics_stage1.score_patient(ct_cal, out_cal, mask_cal)
        else:
            out_cal, ct_cal, mask_cal = process(stage2_out, origin_ct, origin_location, origin_mask)
            global_metrics = synthrad_metrics_stage2.score_patient(ct_cal, out_cal, mask_cal)

        if iteration == slice_index or (save_all and iteration % 10 == 0):
            # 处理和转换图像数据
            show_cbct = origin_cbct.detach().cpu().squeeze().numpy()
            show_origin_ct = origin_ct.detach().cpu().squeeze().numpy()
            mask = origin_mask.detach().cpu().squeeze().numpy()
            show_stage1_out = stage1_out.detach().cpu().squeeze().numpy()
            show_enhanced_ct = enhance_ct.detach().cpu().squeeze().numpy()
            show_stage2_out = stage2_out.detach().cpu().squeeze().numpy() if stage2_out is not None else None

            # 保存图像
            self.save_visualization(epoch, iteration, slice_index, global_metrics, show_cbct, show_origin_ct,
                                    show_stage1_out,
                                    mask, show_enhanced_ct, show_stage2_out)

        metrics[0, 0, iteration] = global_metrics['psnr']
        metrics[0, 1, iteration] = global_metrics['ssim']
        metrics[0, 2, iteration] = global_metrics['mae']

        return metrics

    def test(self, data_loader_test, epoch):
        """封装后的测试函数"""
        self.stage1.eval()
        self.stage2.eval()
        self.resbranch.eval()

        ds_len = len(data_loader_test)
        metrics = np.zeros((1, 3, ds_len))
        save_all = epoch == self.epoch_stage1 * 3
        slice_index = 120

        with torch.no_grad():
            for iteration, (images, image_locations) in enumerate(tqdm(data_loader_test)):

                images = images.to(self.device)
                origin_cbct, origin_ct, enhance_ct, mask = torch.split(images, [5, 1, 1, 1], dim=1)

                stage1_out = self.stage1(origin_cbct * mask)

                if epoch <= self.epoch_stage1:
                    metrics = self.process_output(stage1_out, None, origin_cbct, mask, origin_ct, enhance_ct,
                                                  image_locations,
                                                  epoch, iteration, slice_index, stage=1, metrics=metrics)
                elif epoch <= self.epoch_stage2:
                    stage2_out = self.stage2(stage1_out * mask)
                    metrics = self.process_output(stage1_out, stage2_out, origin_cbct, mask, origin_ct,
                                                  enhance_ct, image_locations,
                                                  epoch, iteration, slice_index, stage=2, metrics=metrics)
                else:
                    stage2_out = self.stage2(stage1_out * mask)
                    stage3_out = self.resbranch(origin_cbct * mask)
                    final_out = stage2_out + stage3_out
                    metrics = self.process_output(final_out, stage2_out, origin_cbct, mask, origin_ct, enhance_ct,
                                                  image_locations, epoch,
                                                  iteration, slice_index, stage=3, metrics=metrics, save_all=save_all)

            # 打印和记录最终测试结果
            avg_psnr = np.nanmean(metrics[0][0])
            avg_ssim = np.nanmean(metrics[0][1])
            avg_mae = np.nanmean(metrics[0][2])

            print(f'psnr: {avg_psnr} ssim: {avg_ssim} mae: {avg_mae}')
            log_str = f"test epoch:{epoch} psnr {avg_psnr} ssim:{avg_ssim} mae:{avg_mae}"
            self.logger.info(log_str)

        return {'psnr': avg_psnr, 'ssim': avg_ssim, 'mae': avg_mae}

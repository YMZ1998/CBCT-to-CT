import numpy as np


def process(out, ct, lct, mask, min_v=-1024, max_v=3000):
    out = out.detach().cpu().squeeze().numpy()
    ct = ct.detach().cpu().squeeze().numpy()
    lct = lct.squeeze().numpy()
    if mask is not None:
        mask = mask.detach().cpu().squeeze().numpy()

    out = (out + 1) / 2
    ct = (ct + 1) / 2
    out = out * (max_v - min_v) + min_v
    ct = ct * (max_v - min_v) + min_v

    # Clip 图像值
    out = np.clip(out, min_v, max_v)
    ct = np.clip(ct, min_v, max_v)

    y_min, x_min = int(lct[1]), int(lct[0])
    y_max, x_max = y_min + int(lct[3]), x_min + int(lct[2])

    # 裁剪出图像区域
    out = out[y_min:y_max, x_min:x_max]
    ct = ct[y_min:y_max, x_min:x_max]
    if mask is not None:
        mask = mask[y_min:y_max, x_min:x_max]
    # print("out", np.min(out), np.max(out))
    # print("ct", np.min(ct), np.max(ct))
    # print("mask", np.min(mask), np.max(mask))
    return out, ct, mask

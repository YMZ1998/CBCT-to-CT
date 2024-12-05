import numpy as np

def process(out, ct, lct, mask, min_v=-1024, max_v=3000):
    out = out.detach().cpu().squeeze().numpy()
    ct = ct.detach().cpu().squeeze().numpy()
    lct = lct.squeeze().numpy()
    mask = mask.detach().cpu().squeeze().numpy()

    # 归一化
    out = out * (max_v - min_v) + min_v
    ct = ct * (max_v - min_v) + min_v

    # Clip 图像值
    out = np.clip(out, min_v, max_v)
    ct = np.clip(ct, min_v, max_v)

    y_min, x_min = lct[1].astype(int), lct[0].astype(int)
    y_max, x_max = y_min + lct[3].astype(int), x_min + lct[2].astype(int)

    # 裁剪出图像区域
    out = out[y_min:y_max, x_min:x_max]
    ct = ct[y_min:y_max, x_min:x_max]
    mask = mask[y_min:y_max, x_min:x_max]

    return out, ct, mask

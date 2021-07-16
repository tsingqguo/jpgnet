import numpy as np
import torch
import os
import cv2
import kpn.utils as kpn_utils

# def save(img, path):
#     img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
#     img_copy = np.clip(img_copy, 0, pixel_max_cnt)
#     img_copy = img_copy.astype(np.uint8)[0, :, :, :]
#     img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
#     if (height != -1) and (width != -1):
#         img_copy = cv2.resize(img_copy, (width, height))

def get_different(kpn, gan, smart, gt, sample_folder, id):
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    kpn = kpn_utils.recover_process(kpn, height=-1, width=-1)
    gan = kpn_utils.recover_process(gan, height=-1, width=-1)
    smart = kpn_utils.recover_process(smart, height=-1, width=-1)
    gt = kpn_utils.recover_process(gt, height=-1, width=-1)

    gap0 = np.abs((kpn - gt))
    gap0 = (gap0 - np.min(gap0)) / np.max(gap0)
    gap0 = (gap0 * 255).astype(np.int)
    save_img_path = os.path.join(sample_folder, '{}_d_kpn.jpg'.format(id))
    cv2.imwrite(save_img_path, gap0)

    gap1 = np.abs((gan - gt))
    gap1 = (gap1-np.min(gap1)) / np.max(gap1)
    gap1 = (gap1 * 255).astype(np.int)
    save_img_path = os.path.join(sample_folder, '{}_d_gan.jpg'.format(id))
    cv2.imwrite(save_img_path, gap1)


    gap2 = np.abs((smart - gt))
    gap2 = (gap2 - np.min(gap2)) / np.max(gap2)
    gap2 = (gap2 * 255).astype(np.int)
    save_img_path = os.path.join(sample_folder, '{}_d_smart.jpg'.format(id))
    cv2.imwrite(save_img_path, gap2)

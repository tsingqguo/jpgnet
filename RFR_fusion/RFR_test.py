import json
import os
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def load_file_list_recursion(fpath, result):
    allfilelist = os.listdir(fpath)
    for file in allfilelist:
        filepath = os.path.join(fpath, file)
        if os.path.isdir(filepath):
            load_file_list_recursion(filepath, result)
        else:
            result.append(filepath)
            # print(len(result))


def npk_val(img_pred_recover, img_gt_recover):
    psnr = compare_psnr(img_gt_recover, img_pred_recover, data_range=255)
    ssim = compare_ssim(img_gt_recover, img_pred_recover, multichannel=True, data_range=255)

    return psnr, ssim


def eval():
    gt = []
    pre = []

    # load_file_list_recursion('J:/cv/DATA/Celeba/256', gt)
    load_file_list_recursion('J:/CV/workspace/final/RFR/results/results', pre)
    load_file_list_recursion('J:/CV/workspace/final/RFR/results/results', gt)

    # load_file_list_recursion('J:/cv/EDGE/farm/gt', gt)
    # load_file_list_recursion('J:/cv/EDGE/farm/20_pre', pre)

    psnr_sum, ssim_sum = 0, 0
    n = 0
    for i in range(len(gt)):
        gt_img = cv2.imread(gt[i])
        pre_img = cv2.imread(pre[i])

        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        psnr, ssim = npk_val(pre_img, gt_img)

        psnr_sum += psnr
        ssim_sum += ssim
        n += 1
        print('psnr:{}  ssim:{}'.format(psnr, ssim))

    print('psnr_ave:{}  ssim_ave:{}'.format(psnr_sum/n, ssim_sum/n))

eval()









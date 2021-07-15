import argparse

import torch
# from skimage.measure import compare_ssim, compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import kpn.dataset as dataset
import kpn.utils as utils
from kpn.config import get_opt
import numpy as np
import cv2
import random

cur_name = ''

def get_uncertainty(core, mask):
    mask = mask.data.cpu().numpy()[0][0].astype(np.uint8)

    core = core.view(1, 3, -1, 256, 256)
    ave = core.sum(dim=2) / core.size(2)
    ave = ave[0].data.cpu().numpy()

    ave = np.clip(ave * 255, 0, 255).astype(np.uint8)
    # merge = ave[2] + ave[1] + ave[0]
    merge = ave[2] + ave[0]
    # merge = ave[1]

    merge[merge < 220] = 0.0
    merge[merge >= 220] = 1.0
    #
    # merge = cv2.dilate(merge, np.ones((3, 3), np.uint8), iterations=1)
    # merge = cv2.erode(merge, np.ones((3, 3), np.uint8), iterations=1)

    # merge = cv2.erode(mask, np.ones((6, 6), np.uint8), iterations=1)

    return merge



def npk_val(pre, gt):
    img_pred_recover = utils.recover_process(pre, height=-1, width=-1)
    img_gt_recover = utils.recover_process(gt, height=-1, width=-1)

    psnr = utils.psnr(img_pred_recover, img_gt_recover)
    ssim = compare_ssim(img_gt_recover, img_pred_recover, multichannel=True, data_range=255)

    return psnr, ssim

def save_sample(list_img, list_name, s_name, dir, h_origin, w_origin):
    utils.save_sample_png(sample_folder=dir, sample_name=s_name, img_list=list_img,
                          name_list=list_name, pixel_max_cnt=255, height=h_origin, width=w_origin)



if __name__ == "__main__":
    opt = get_opt()
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    if opt.no_gpu:
        generator = utils.create_generator(opt)
    else:
        generator = utils.create_generator(opt).cuda()

    '''
    parm={}
    for name,parameters in generator.named_parameters():
        print(name,':',parameters.size())
        parm[name]=parameters.detach().cpu().numpy()
    print(parm['conv_final.weight'])
    print(parm['conv_final.bias'])
    '''

    test_dataset = dataset.DenoisingValDataset(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = opt.save_name
    utils.check_path(sample_folder)

    psnr_sum, psnr_ave, ssim_sum, ssim_ave, eval_cnt = 0, 0, 0, 0, 0
    
    # forward
    for i, (true_input, true_target, height_origin, width_origin) in enumerate(test_loader):

        # To device
        if opt.no_gpu:
            true_input = true_input
            true_target = true_target
        else:
            true_input = true_input.cuda()
            true_target = true_target.cuda()            

        # Forward propagation
        with torch.no_grad():
            #print(true_input.size()) 
            fake_target = generator(true_input, true_input)
        
        #print(fake_target.shape, true_input.shape)

        # Save
        print('The %d-th iteration' % (i))
        img_list = [true_input, fake_target, true_target]
        name_list = ['in', 'pred', 'gt']
        sample_name = '%d' % (i+1)
        utils.save_sample_png(sample_folder = sample_folder, sample_name = '%d' % (i + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255, height = height_origin, width = width_origin)
        
        # Evaluation
        #psnr_sum = psnr_sum + utils.psnr(cv2.imread(sample_folder + '/' + sample_name + '_' + name_list[1] + '.png').astype(np.float32), cv2.imread(sample_folder + '/' + sample_name + '_' + name_list[2] + '.png').astype(np.float32))
        img_pred_recover = utils.recover_process(fake_target, height = height_origin, width = width_origin)
        img_gt_recover = utils.recover_process(true_target, height = height_origin, width = width_origin)
        #psnr_sum = psnr_sum + utils.psnr(utils.recover_process(fake_target, height = height_origin, width = width_origin), utils.recover_process(true_target, height = height_origin, width = width_origin))
        psnr_sum = psnr_sum + utils.psnr(img_pred_recover, img_gt_recover)
        ssim_sum = ssim_sum + compare_ssim(img_gt_recover, img_pred_recover, multichannel = True, data_range = 255) 
        eval_cnt = eval_cnt + 1
        
    psnr_ave = psnr_sum / eval_cnt
    ssim_ave = ssim_sum / eval_cnt
    psnr_file = "./data/psnr_data.txt"
    ssim_file = "./data/ssim_data.txt"
    psnr_content = opt.load_name + ": " + str(psnr_ave) + "\n"
    ssim_content = opt.load_name + ": " + str(ssim_ave) + "\n"
    utils.text_save(content = psnr_content, filename = psnr_file)
    utils.text_save(content = ssim_content, filename = ssim_file)
    
    

import argparse


def str2bool(v):
    #print(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_opt():
    import torch
    # from skimage.measure import compare_ssim, compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim

    import kpn.dataset as dataset
    import kpn.utils as utils

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()

    # parser.add_argument('--load_name', type=str, default='./model/v3_rain100H.pth', help='load the pre-trained model with certain epoch')
    parser.add_argument('--kpn1_model', type=str, default='./checkpoints/1_15_KPN_bs_1_.pth', help='load pre')
    parser.add_argument('--kpn2_model', type=str, default='./checkpoints/1_kpn_dunhuang_edge_smart.pth', help='load pre')

    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--data_eval', type=str, default='./gan/data/eval', help='images baseroot')
    parser.add_argument('--data_train', type=str, default='./gan/data/train', help='images baseroot')
    parser.add_argument('--data_mask', type=str, default='./gan/data/mask', help='images baseroot')

    parser.add_argument('--save_by_iter', type=int, default=1,
                        help='interval between model checkpoints (by iterations)')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Adam: learning rate for G / D')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for optimizer')

    parser.add_argument('--train_batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--test_batch_size', type=int, default=1, help='size of the batches')

    parser.add_argument('--eval_interval', type=int, default=1, help='size of the batches')
    parser.add_argument('--train_sample_interval', type=int, default=1, help='size of the batches')
    parser.add_argument('--eval_sample_interval', type=int, default=1, help='size of the batches')

    parser.add_argument('--test_sample', type=str, default='./result/test_sample', help='saving path that is a folder')
    parser.add_argument('--train_sample', type=str, default='./result/train_sample', help='saving path that is a folder')
    parser.add_argument('--eval_sample', type=str, default='./result/eval_sample', help='saving path that is a folder')
    parser.add_argument('--save_model', type=str, default='./result/model', help='saving path that is a folder')



    # Initialization parameters
    parser.add_argument('--color', type=str2bool, default=True, help='input type')
    parser.add_argument('--burst_length', type=int, default=1, help='number of photos used in burst setting')
    parser.add_argument('--blind_est', type=str2bool, default=True, help='variance map')
    parser.add_argument('--kernel_size', type=list, default=[3], help='kernel size')
    parser.add_argument('--sep_conv', type=str2bool, default=False, help='simple output type')
    parser.add_argument('--channel_att', type=str2bool, default=False, help='channel wise attention')
    parser.add_argument('--spatial_att', type=str2bool, default=False, help='spatial wise attention')
    parser.add_argument('--upMode', type=str, default='bilinear', help='upMode')
    parser.add_argument('--core_bias', type=str2bool, default=False, help='core_bias')
    parser.add_argument('--init_type', type=str, default='xavier', help='initialization type of generator')
    parser.add_argument('--init_gain', type=float, default=0.02, help='initialization gain of generator')
    # Dataset parameters

    parser.add_argument('--crop', type=str2bool, default=False, help='whether to crop input images')
    parser.add_argument('--crop_size', type=int, default=512, help='single patch size')
    parser.add_argument('--geometry_aug', type=str2bool, default=False, help='geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type=str2bool, default=False, help='geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type=float, default=1, help='min scaling factor')
    parser.add_argument('--scale_max', type=float, default=1, help='max scaling factor')
    parser.add_argument('--add_noise', type=str2bool, default=False, help='whether to add noise to input images')
    parser.add_argument('--mu', type=int, default=0, help='Gaussian noise mean')
    parser.add_argument('--sigma', type=int, default=30, help='Gaussian noise variance: 30 | 50 | 70')
    opt = parser.parse_args()

    return opt
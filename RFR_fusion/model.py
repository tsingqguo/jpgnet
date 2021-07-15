import torch
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time
from temp import map
from PIL import  Image
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import kpn.utils as kpn_utils
from kpn.config import get_opt
from kpn.validation import npk_val, get_uncertainty, get_uncertainty_2
import config
import kpn.pytorch_ssim as pytorch_ssim
from different import get_different
import time
import torchvision
import lpips


class RFRNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
    
    def initialize_model(self, path=None, train=True):
        self.G = RFRNet()
        self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            print('RFR is loaded.........Model Initialized')
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
                print('iter: ', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0
        
    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")
        
    def train(self, train_loader, eval_loader, save_path, finetune = False, iters=450000):
        self.G.train(finetune = finetune)

        if finetune:
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = 5e-5)
        print("Starting training from iteration:{:d}".format(self.iter))

        s_time = time.time()
        while self.iter < iters:
            for items in train_loader:

                self.G.train()
                for para in self.G.parameters():
                    para.requires_grad = True


                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks

                masks = torch.cat([masks] * 3, dim=1)

                self.forward(masked_images, masks, gt_images)
                self.update_parameters()
                self.iter += 1
                
                if self.iter % config.log_interval == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f" %(self.iter, self.l1_loss_val/50, int_time))
                    s_time = time.time()
                    self.l1_loss_val = 0.0

                if self.iter % config.eval_interval == 0:
                    print('start eval-------------------------------')
                    self.eval(eval_loader)
                
                if self.iter % config.save_interval == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    save_ckpt('{:s}/g_{:d}.pth'.format(save_path, self.iter ), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)

        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/g_{:s}.pth'.format(save_path, "final"), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
    def test(self, test_loader, result_save_path):

        opt = get_opt()
        generator = kpn_utils.create_generator(opt)
        # -----------------kpn--------------------
        if torch.cuda.is_available():
            generator = generator.cuda()
        # ------------------------------------------
        generator.eval()

        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False
        count = 0

        psnr_list = []
        ssim_list = []
        kpn_list_psnr = []
        kpn_list_ssim = []
        final_list_psnr = []
        final_list_ssim = []

        for items in test_loader:
            gt_images, masks,              img_rainy_kpn, img_gt_kpn, H, W = self.__cuda__(*items)
            masked_images = gt_images * masks
            masks = torch.cat([masks]*3, dim = 1)
            fake_B, mask = self.G(masked_images, masks)
            comp_B = fake_B * (1 - masks) + gt_images * masks

            #------------------------metric
            psnr, ssim = self.metric(gt_images.clone(), comp_B.clone())
            psnr_list.append(psnr)
            ssim_list.append(ssim)


            #-------------------------kpn
            kpn_out, core = generator(img_rainy_kpn, img_rainy_kpn)
            kpn_out = (kpn_out * (1-masks)) + (img_gt_kpn * masks)

            kpn_pnsr, kpn_pnsr2, kpn_ssim, kpn_out_r = npk_val(kpn_out, img_gt_kpn, H, W, True)
            kpn_list_psnr.append(kpn_pnsr)
            kpn_list_ssim.append(kpn_ssim)


            #-------------------------final
            uncertainty = get_uncertainty(core, 1-masks)

            if len(final_list_ssim) % 1 == 0:
                Image.fromarray(uncertainty * 255).save(
                    os.path.join(opt.save_name, '{}_un.png'.format(map['name'])))

            uncertainty = torch.from_numpy(uncertainty).float()

            if torch.cuda.is_available():
                uncertainty = uncertainty.cuda()

            # final_out = self.postprocess(outputs_merged)
            final_pre = kpn_out * (1 - uncertainty) + comp_B * uncertainty
            final_pnsr, final_pnsr2, final_ssim, _ = npk_val(final_pre, img_gt_kpn, H, W, True)

            final_list_psnr.append(final_pnsr)
            final_list_ssim.append(final_ssim)

            print("psnr:{}/{}  ssim:{}/{}  kpn_psnr:{}/{}  kpn_ssim:{}/{}  final_psnr:{}/{} final_ssim:{}/{}       {}".format(
                psnr, np.average(psnr_list), ssim, np.average(ssim_list),
                kpn_pnsr, np.average(kpn_list_psnr), kpn_ssim, np.average(kpn_list_ssim),
                final_pnsr, np.average(final_list_psnr), final_ssim, np.average(final_list_ssim),
                len(ssim_list)))

            img_list = [img_rainy_kpn, img_gt_kpn, comp_B, kpn_out, final_pre]
            name_list = ['in', 'gt', 'rfr', 'kpn', 'final']

            # img_list = [final_pre]
            # name_list = ['']

            if len(kpn_list_ssim) % 1 == 0:
                kpn_utils.save_sample_png(sample_folder=opt.save_name, sample_name=map['name'],
                                          img_list=img_list,
                                          name_list=name_list, pixel_max_cnt=255, height=H,
                                          width=W)

            #----------------------save
            # if len(ssim_list) % 20 == 0:
            #     if not os.path.exists('{:s}/results'.format(result_save_path)):
            #         os.makedirs('{:s}/results'.format(result_save_path))
            #
            #     file_path = '{}/results/{}'.format(result_save_path, map['name'])
            #
            #     comp_B = comp_B.clamp_(0, 1) * 255.0
            #     comp_B = comp_B.permute(0, 2, 3, 1)
            #     comp_B = Image.fromarray(comp_B.cpu().detach().numpy().astype(np.uint8).squeeze())
            #     comp_B.save(file_path)


            # for k in range(comp_B.size(0)):
            #     count += 1
            #     grid = make_grid(comp_B[k:k+1])
            #     file_path = '{:s}/results/{:s}'.format(result_save_path, map['name'])
            #     save_image(grid, file_path)
            #
            #     # grid = make_grid(gt_images[k:k+1])
            #     # file_path = '{:s}/results/gt_img_{:d}.png'.format(result_save_path, count)
            #     # save_image(grid, file_path)

        print('rfr_psnr_ave:{} rfr_ssim_ave:{} kpn_ave_psnr:{} kpn_ave_ssim:{}  final_ave_psnr:{} final_ssim_ssim:{}'.format(
            np.average(psnr_list), np.average(ssim_list),
            np.average(kpn_list_psnr), np.average(kpn_list_psnr),
            np.average(final_list_psnr), np.average(final_list_ssim)
        ))


    def eval(self, eval_loader):

        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False

        psnr_list = []
        ssim_list = []
        i = 0
        for items in eval_loader:
            gt_images, masks = self.__cuda__(*items)
            masked_images = gt_images * masks
            masks = torch.cat([masks] * 3, dim=1)

            fake_B, mask = self.G(masked_images, masks)
            comp_B = fake_B * (1 - masks) + gt_images * masks

            # ------------------------metric
            psnr, ssim = self.metric(gt_images.clone(), comp_B.clone())
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            print(
                "psnr:{}/{}  ssim:{}/{} {}".format(
                    psnr, np.average(psnr_list), ssim, np.average(ssim_list), len(ssim_list)))

            img_list = [masked_images, gt_images, comp_B, fake_B]
            name_list = ['in', 'gt', 'rfr', 'fake']


            if len(ssim_list) % config.eval_sample == 0:
                kpn_utils.save_sample_png(sample_folder=config.eval_save_path, sample_name='ite_{}_{}'.format(self.iter, i),
                                          img_list=img_list,
                                          name_list=name_list, pixel_max_cnt=255, height=-1,
                                          width=-1)
            i = i+1

            if len(ssim_list) == config.eval_count:
                break

        print('ite:{} rfr_psnr_ave:{} rfr_ssim_ave:{}'.format(self.iter, np.average(psnr_list), np.average(ssim_list)))


    def fusion_train(self, train_loader, eval_loader, save_path, finetune = False, iters=450000):
        opt = get_opt()
        generator1 = kpn_utils.create_generator(opt, 1, opt.kpn1_model)
        generator2 = kpn_utils.create_generator(opt, 2, opt.kpn2_model)

        criterion_L1 = torch.nn.L1Loss()
        criterion_ssim = pytorch_ssim.SSIM()

        # -----------------kpn--------------------
        if torch.cuda.is_available():
            generator1 = generator1.cuda()
            generator2 = generator2.cuda()
            criterion_L1 = criterion_L1.cuda()
            criterion_ssim = criterion_ssim.cuda()

        optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, generator2.parameters()), lr=opt.lr_g,
                                       betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
        # ------------------------------------------
        generator1.eval()
        generator2.train()
        self.G.eval()

        for para in self.G.parameters():
            para.requires_grad = False

        while True:
            for items in train_loader:

                generator2.iteration += 1

                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                masks_3 = torch.cat([masks] * 3, dim=1)

                fake_B, mask = self.G(masked_images, masks_3)
                comp_B = fake_B * (1 - masks_3) + gt_images * masks_3
                res1 = comp_B.detach()

                # kpn model
                img_rainy_kpn = gt_images * masks
                kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                kpn_out = kpn_out * (1-masks) + gt_images * masks
                res2 = kpn_out.detach()
                uncertainty_map = core.detach()

                # kpn2
                uMap = get_uncertainty(uncertainty_map, gt_images.size(0))
                input2 = torch.cat((res1, res2, uncertainty_map, uMap), dim=1)
                input2_ = torch.cat((res1, res2), dim=1)
                kpn_out2, _ = generator2(input2, input2_)

                ssim_loss = -criterion_ssim(gt_images, kpn_out2)
                Pixellevel_L1_Loss = criterion_L1(gt_images, kpn_out2)

                loss = Pixellevel_L1_Loss + 0.2 * ssim_loss

                optimizer_G.zero_grad()

                loss.backward()
                optimizer_G.step()

                # save
                if generator2.iteration % opt.save_by_iter == 0:
                    self.save_model(opt, generator2.iteration, generator2)

                # Learning rate decrease at certain epochs
                # self.adjust_learning_rate(opt, (epoch + 1), optimizer_G)

                #sample
                if generator2.iteration % opt.train_sample_interval == 0:
                    masks_ = torch.cat([1-masks] * 3, dim=1)
                    img_list = [img_rainy_kpn, masks_, kpn_out2, res1, res2, gt_images]
                    name_list = ['in', 'mask', 'pred', 'edge_out', 'kpn_out', 'gt']
                    kpn_utils.save_sample_png(sample_folder=opt.train_sample, sample_name='ite_{}'.format(generator2.iteration),
                                          img_list=img_list, name_list=name_list, pixel_max_cnt=255)

                # eval
                if generator2.iteration % opt.eval_interval == 0:
                    generator2.eval()
                    cur_psnr, cur_ssim = self.fusion_eval(eval_loader, generator1, generator2, opt)
                    generator2.train()

                print('iteration:{}  ssim_loss:{}  l1_loss:{}  loss:{}'.format(generator2.iteration, ssim_loss.item(),
                                                                               Pixellevel_L1_Loss.item(), loss.item()))


    def fusion_eval(self, eval_loader, generator1, generator2, opt):

        psnr_list = []
        ssim_list = []
        for items in eval_loader:
            with torch.no_grad():
                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                masks_3 = torch.cat([masks] * 3, dim=1)

                fake_B, mask = self.G(masked_images, masks_3)
                comp_B = fake_B * (1 - masks_3) + gt_images * masks_3
                res1 = comp_B.detach()

                # kpn model
                img_rainy_kpn = gt_images * masks
                kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                kpn_out = kpn_out * (1-masks) + gt_images * masks
                res2 = kpn_out.detach()
                uncertainty_map = core.detach()

                # kpn2
                uMap = get_uncertainty(uncertainty_map, gt_images.size(0))
                input2 = torch.cat((res1, res2, uncertainty_map, uMap), dim=1)
                input2_ = torch.cat((res1, res2), dim=1)
                kpn_out2, _ = generator2(input2, input2_)
                kpn_out2_merged = kpn_out2 * (1-masks) + gt_images * masks

                # sample
                if len(psnr_list) % opt.eval_sample_interval == 0:
                    masks_ = torch.cat([1-masks] * 3, dim=1)
                    img_list = [img_rainy_kpn, masks_, kpn_out2_merged, res1, res2, gt_images]
                    name_list = ['in', 'mask', 'pred', 'edge_out', 'kpn_out', 'gt']
                    kpn_utils.save_sample_png(sample_folder=opt.eval_sample,
                                              sample_name='ite_{}_{}'.format(generator2.iteration, len(psnr_list)),
                                              img_list=img_list, name_list=name_list, pixel_max_cnt=255)

                img_pred = kpn_utils.recover_process(kpn_out2_merged, height=-1, width=-1)
                img_gt = kpn_utils.recover_process(gt_images, height=-1, width=-1)

                psnr = kpn_utils.psnr(img_pred, img_gt)
                ssim = compare_ssim(img_gt, img_pred, multichannel=True, data_range=255)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            print('iteration {}_{}  psnr:{}/{}  ssim:{}/{}'.format(generator2.iteration, len(psnr_list),
                                                                   psnr, np.average(psnr_list),
                                                                   ssim, np.average(ssim_list)))

            if len(psnr_list) >= 1000:
                break

        ave_psnr = np.average(psnr_list)
        ave_ssim = np.average(ssim_list)
        print('iteration:{} psnr_ave:{} ssim_ave:{}'.format(generator2.iteration, ave_psnr, ave_ssim))

        return ave_psnr, ave_ssim

    def fusion_test(self, test_loader):
        opt = get_opt()
        generator1 = kpn_utils.create_generator(opt, 1, opt.kpn1_model)
        generator2 = kpn_utils.create_generator(opt, 2, opt.kpn2_model)

        transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        loss_fn_vgg = lpips.LPIPS(net='vgg')

        # -----------------kpn--------------------
        if torch.cuda.is_available():
            generator1 = generator1.cuda()
            generator2 = generator2.cuda()
            loss_fn_vgg = loss_fn_vgg.cuda()

        generator1.eval()
        generator2.eval()
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False

        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []

        kpn_psnr_list = []
        kpn_ssim_list = []

        binary_psnr_list = []
        binary_ssim_list = []
        binary_l1_list = []
        binary_lpips_list = []

        rfr_psnr_list = []
        rfr_ssim_list = []
        rfr_l1_list = []
        rfr_lpips_list = []

        input_psnr_list = []
        input_ssim_list = []

        kpn1_time = []
        gan_time = []
        smart_time = []
        binary_time = []

        for i in range(1):
            for items in test_loader:
                with torch.no_grad():
                    gt_images, masks = self.__cuda__(*items)
                    masked_images = gt_images * masks
                    masks_3 = torch.cat([masks] * 3, dim=1)


                    t1 = time.time()

                    fake_B, mask = self.G(masked_images, masks_3)
                    comp_B = fake_B * (1 - masks_3) + gt_images * masks_3
                    gan_time.append(time.time() - t1)
                    res1 = comp_B.detach()



                    rfr_psnr, rfr_ssim = npk_val(res1, gt_images)
                    rfr_psnr_list.append(rfr_psnr)
                    rfr_ssim_list.append(rfr_ssim)

                    rfr_l1_loss = torch.nn.functional.l1_loss(res1, gt_images, reduction='mean').item()
                    rfr_l1_list.append(rfr_l1_loss)

                    if torch.cuda.is_available():
                        rfr_pl = loss_fn_vgg(transf(res1[0].cpu()).cuda(), transf(gt_images[0].cpu()).cuda()).item()
                        rfr_lpips_list.append(rfr_pl)
                    else:
                        rfr_pl = loss_fn_vgg(transf(res1[0].cpu()), transf(gt_images[0].cpu())).item()
                        rfr_lpips_list.append(rfr_pl)

                    # kpn model-----------------------------------
                    img_rainy_kpn = gt_images * masks
                    t1 = time.time()
                    kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                    kpn_out_merged = kpn_out * (1 - masks) + gt_images * masks
                    kpn1_time.append(time.time() - t1)
                    res2 = kpn_out_merged.detach()
                    uncertainty_map = core.detach()


                    # kpn2------------------
                    uMap = get_uncertainty(uncertainty_map, gt_images.size(0))

                    input2 = torch.cat((res1, res2, uncertainty_map, uMap), dim=1)
                    input2_ = torch.cat((res1, res2), dim=1)
                    t1 = time.time()
                    kpn_out2, _ = generator2(input2, input2_)
                    kpn_out2_merged = kpn_out2 * (1 - masks) + gt_images * masks
                    smart_time.append(time.time() - t1)

                    # binary fusion--------------------

                    uncertainty = get_uncertainty_2(uncertainty_map, gt_images.size(0))
                    # uncertainty = torch.from_numpy(uncertainty).float()
                    # if torch.cuda.is_available():
                    #     uncertainty = uncertainty.cuda()

                    t1 = time.time()
                    binary_pre = res2 * (1 - uncertainty) + res1 * uncertainty
                    binary_time.append(time.time() - t1)
                    binary_psnr, binary_ssim = npk_val(binary_pre, gt_images)

                    binary_psnr_list.append(binary_psnr)
                    binary_ssim_list.append(binary_ssim)

                    binary_l1_loss = torch.nn.functional.l1_loss(binary_pre, gt_images, reduction='mean').item()
                    binary_l1_list.append(binary_l1_loss)

                    if torch.cuda.is_available():
                        binary_pl = loss_fn_vgg(transf(binary_pre[0].cpu()).cuda(), transf(gt_images[0].cpu()).cuda()).item()
                        binary_lpips_list.append(binary_pl)
                    else:
                        binary_pl = loss_fn_vgg(transf(binary_pre[0].cpu()),
                                                transf(gt_images[0].cpu())).item()
                        binary_lpips_list.append(binary_pl)


                    # kpn--------------------------------
                    kpn_psnr, kpn_ssim = npk_val(res2, gt_images)
                    kpn_psnr_list.append(kpn_psnr)
                    kpn_ssim_list.append(kpn_ssim)

                    # input gt-------------------------------------
                    input_psnr, input_ssim = npk_val(img_rainy_kpn, gt_images)
                    input_psnr_list.append(input_psnr)
                    input_ssim_list.append(input_ssim)


                    # sample
                    if len(psnr_list) % 1 == 0:
                        # different-------------------------------------
                        # get_different(kpn_out, fake_B, kpn_out2, gt_images, opt.test_sample, len(psnr_list))
                        # masks_ = torch.cat([1 - masks] * 3, dim=1)
                        # un_map = uncertainty.unsqueeze(0).unsqueeze(0)
                        # un_map = torch.cat([un_map] * 3, dim=1)
                        # img_list = [img_rainy_kpn, masks_, kpn_out2_merged, res1, res2, gt_images, un_map]
                        # name_list = ['in', 'mask', 'pred', 'edge_out', 'kpn_out', 'gt', 'un_map']
                        # kpn_utils.save_sample_png(sample_folder=opt.test_sample,
                        #                           sample_name='{}_'.format(len(psnr_list)),
                        #                           img_list=img_list, name_list=name_list, pixel_max_cnt=255)

                        masks_ = torch.cat([1 - masks] * 3, dim=1)
                        un_map = uncertainty.unsqueeze(0).unsqueeze(0)
                        uncertainty = torch.cat([uncertainty] * 3, dim=1)
                        img_list = [kpn_out2_merged, res1, gt_images, uncertainty, res2, binary_pre]
                        name_list = ['pred', 'edge_out', 'gt', 'umap', 'kpn', 'binary']
                        kpn_utils.save_sample_png(sample_folder=opt.test_sample,
                                                  sample_name='{}_'.format(len(psnr_list)),
                                                  img_list=img_list, name_list=name_list, pixel_max_cnt=255)

                    img_pred = kpn_utils.recover_process(kpn_out2_merged, height=-1, width=-1)
                    img_gt = kpn_utils.recover_process(gt_images, height=-1, width=-1)

                    psnr = kpn_utils.psnr(img_pred, img_gt)
                    ssim = compare_ssim(img_gt, img_pred, multichannel=True, data_range=255)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    l1_loss = torch.nn.functional.l1_loss(kpn_out2_merged, gt_images, reduction='mean').item()
                    l1_list.append(l1_loss)

                    if torch.cuda.is_available():
                        pl = loss_fn_vgg(transf(kpn_out2_merged[0].cpu()).cuda(), transf(gt_images[0].cpu()).cuda()).item()
                        lpips_list.append(pl)
                    else:
                        pl = loss_fn_vgg(transf(kpn_out2_merged[0].cpu()),
                                         transf(gt_images[0].cpu())).item()
                        lpips_list.append(pl)

                print(
                    '{}  psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  kpn_psnr:{}/{}  kpn_ssim:{}/{}  '
                    'binary_psnr:{}/{}  binary_ssim:{}/{} binary_l1:{}/{}  binary_lpips:{}/{} '
                    'rfr_psnr:{}/{}  rfr_ssim:{}/{} rfr_l1:{}/{}  rfr_lpips:{}/{}'
                    'input_psnr:{}/{}  input_ssim:{}/{}'.format(
                        len(psnr_list),
                        psnr, np.average(psnr_list), ssim, np.average(ssim_list), l1_loss, np.average(l1_list), pl, np.average(lpips_list),
                        kpn_psnr, np.average(kpn_psnr_list), kpn_ssim, np.average(kpn_ssim_list),
                        binary_psnr, np.average(binary_psnr_list), binary_ssim, np.average(binary_ssim_list), binary_l1_loss, np.average(binary_l1_list), binary_pl, np.average(binary_lpips_list),
                        rfr_psnr, np.average(rfr_psnr_list), rfr_ssim, np.average(rfr_ssim_list), rfr_l1_loss, np.average(rfr_l1_list), rfr_pl, np.average(rfr_lpips_list),
                        input_psnr, np.average(input_psnr_list), input_ssim, np.average(input_ssim_list))
                )

                if len(smart_time) >= 20000:
                    break

        print('final psnr:{}  ssim:{} l1:{} lpips:{}  kpn_psnr:{}  kpn_ssim:{}  binary_psnr:{}  binary_ssim:{} binary_l1:{}  binary_lpips:{}'
              '  rfr_psnr:{}  rfr_ssim:{} rfr_l1:{}  rfr_lpips:{}   input_psnr:{}  input_ssim:{}'.format(
            np.average(psnr_list), np.average(ssim_list), np.average(l1_list), np.average(lpips_list),
            np.average(kpn_psnr_list), np.average(kpn_ssim_list),
            np.average(binary_psnr_list), np.average(binary_ssim_list), np.average(binary_l1_list), np.average(binary_lpips_list),
            np.average(rfr_psnr_list), np.average(rfr_ssim_list), np.average(rfr_l1_list), np.average(rfr_lpips_list),
            np.average(input_psnr_list), np.average(input_ssim_list))
        )

        print('kpn1:{}  binary:{}  smart:{}  gan:{}'.format(np.average(kpn1_time),
                                                            np.average(binary_time),
                                                            np.average(smart_time),
                                                            np.average(gan_time),))

    def save_model(self, opt, iteration, generator):
        model_name = '{}_KPN_bs_{}.pth'.format(iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_model)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        save_model_path = os.path.join(save_model_path, model_name)

        if torch.cuda.is_available():
            torch.save(generator.state_dict(), save_model_path)
            print('The trained model is successfully saved at iteration {}'.format(iteration))
        else:
            torch.save(generator.state_dict(), save_model_path)
            print('The trained model is successfully saved at iteration {}'.format(iteration))

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.cpu().detach().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

        return psnr, ssim

    
    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
    
    def update_parameters(self):
        self.update_G()
        self.update_D()
    
    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()
    
    def update_D(self):
        return
    
    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        
        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))
        
        loss_G = (  tv_loss * 0.1
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 1
                  + hole_loss * 6)
        
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G
    
    def l1_loss(self, f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)
    
    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
            
    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
            
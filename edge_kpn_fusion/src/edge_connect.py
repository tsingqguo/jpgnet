import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import kpn.utils as kpn_utils
from kpn.config import get_opt
from kpn.validation import npk_val, get_uncertainty
from different import get_different
import torch.nn.functional as F
import lpips
import torchvision.transforms

import kpn.pytorch_ssim as pytorch_ssim
import time

class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=False)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

            print('train datasize:{}   eval datasize:{}'.format(len(self.train_dataset), len(self.val_dataset)))

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load('edge')
            self.inpaint_model.load('paint')

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.edge_model.train()
                self.inpaint_model.train()

                images, images_gray, edges, masks = self.cuda(*items)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                    # metrics
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))
                    logs.append(('gen_loss', gen_loss.item()))

                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration


                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks)
                        outputs = outputs * masks + edges * (1 - masks)
                    else:
                        outputs = edges

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # joint model
                else:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                # progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                print(logs)

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        psnr_all = []
        ssim_all = []

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)
                # metrics
                # precision, recall = self.edgeacc(edges * masks, outputs * masks)
                # logs.append(('precision', precision.item()))
                # logs.append(('recall', recall.item()))
                # logs.append(('gen_loss', gen_loss.item()))

                psnr, ssim = self.metric(edges, outputs)
                psnr_all.append(psnr)
                ssim_all.append(ssim)

            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                # psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                # mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                # logs.append(('psnr', psnr.item()))
                # logs.append(('mae', mae.item()))

                psnr, ssim = self.metric(images, outputs_merged)
                psnr_all.append(psnr)
                ssim_all.append(ssim)


            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs


            print('psnr:{}/{}  ssim:{}/{}'.format(psnr, np.average(psnr_all), ssim, np.average(ssim_all)))
            # progbar.add(len(images), values=logs)

        print('ave_psnr:{}  ave_ssim:{}'.format(
            np.average(psnr_all),
            np.average(ssim_all)
        ))

    def test(self):
        opt = get_opt()
        generator = kpn_utils.create_generator(opt)
        # -----------------kpn--------------------
        if torch.cuda.is_available():
            generator = generator.cuda()
        # ------------------------------------------
        generator.eval()



        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        psnr_list = []
        ssim_list = []
        kpn_list_psnr = []
        kpn_list_ssim = []
        final_list_psnr = []
        final_list_ssim = []


        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks,       img_rainy_kpn, img_gt_kpn, H, W = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))


            pnsr, ssim = npk_val(outputs_merged, images, H, W, True)
            psnr_list.append(pnsr)
            ssim_list.append(ssim)

            # -------------------------kpn
            kpn_out, core = generator(img_rainy_kpn, img_rainy_kpn)
            kpn_out = (kpn_out * masks) + (img_gt_kpn * (1-masks))

            kpn_pnsr, kpn_ssim = npk_val(kpn_out, img_gt_kpn, H, W, True)
            kpn_list_psnr.append(kpn_pnsr)
            kpn_list_ssim.append(kpn_ssim)

            # -------------------------final
            uncertainty = get_uncertainty(core, masks)

            if len(final_list_ssim) % 1 == 0:
                Image.fromarray(uncertainty * 255).save(
                    os.path.join(opt.save_name, '{}_un.png'.format(name)))

            uncertainty = torch.from_numpy(uncertainty).float()

            if torch.cuda.is_available():
                uncertainty = uncertainty.cuda()

            final_pre = kpn_out * (1 - uncertainty) + outputs_merged * uncertainty
            final_pnsr, final_ssim = npk_val(final_pre, img_gt_kpn, H, W, True)

            final_list_psnr.append(final_pnsr)
            final_list_ssim.append(final_ssim)

            print(
                "psnr:{}/{}  ssim:{}/{}  kpn_psnr:{}/{}  kpn_ssim:{}/{}  final_psnr:{}/{} final_ssim:{}/{}       {}".format(
                    pnsr, np.average(psnr_list), ssim, np.average(ssim_list),
                    kpn_pnsr, np.average(kpn_list_psnr), kpn_ssim, np.average(kpn_list_ssim),
                    final_pnsr, np.average(final_list_psnr), final_ssim, np.average(final_list_ssim),
                    len(ssim_list)))

            img_list = [img_rainy_kpn, img_gt_kpn, outputs_merged, kpn_out, final_pre]
            name_list = ['in', 'gt', 'edge', 'kpn', 'final']

            # img_list = [final_pre]
            # name_list = ['']

            if len(kpn_list_ssim) % 1 == 0:
                kpn_utils.save_sample_png(sample_folder=opt.save_name, sample_name=name,
                                          img_list=img_list,
                                          name_list=name_list, pixel_max_cnt=255, height=H,
                                          width=W)




            #--------------------------------metric
            # pre = output.cpu().numpy().astype(np.uint8)
            # gt = self.postprocess(images)[0].cpu().numpy().astype(np.uint8)
            #
            # psnr = min(100, compare_psnr(gt, pre))
            # ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)
            # psnr_list.append(psnr)
            # ssim_list.append(ssim)

            # print("psnr:{}/{}  ssim:{}/{}       {}".format(psnr, np.average(psnr_list),  ssim, np.average(ssim_list), len(ssim_list)))
            #
            # if len(ssim_list) % 20 == 0:
            #     path = os.path.join(self.results_path, name)
            #     imsave(output, path)

            # if self.debug:
            #     edges = self.postprocess(1 - edges)[0]
            #     masked = self.postprocess(images * (1 - masks) + masks)[0]
            #     fname, fext = name.split('.')
            #
            #     imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
            #     imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print(
            'edge_psnr_ave:{} edge_ssim_ave:{} kpn_ave_psnr:{} kpn_ave_ssim:{}  final_ave_psnr:{} final_ssim_ssim:{}'.format(
                np.average(psnr_list), np.average(ssim_list),
                np.average(kpn_list_psnr), np.average(kpn_list_psnr),
                np.average(final_list_psnr), np.average(final_list_ssim)
            ))

    def npn_edge_fusion_train(self):
        opt = get_opt()
        generator1 = kpn_utils.create_generator(opt, 1, opt.kpn1_model)
        generator2 = kpn_utils.create_generator(opt, 2, opt.kpn2_model)

        criterion_L1 = torch.nn.L1Loss()
        criterion_L2 = torch.nn.MSELoss()
        criterion_ssim = pytorch_ssim.SSIM()

        # -----------------kpn--------------------
        if torch.cuda.is_available():
            generator1 = generator1.cuda()
            generator2 = generator2.cuda()
            criterion_L1 = criterion_L1.cuda()
            criterion_L2 = criterion_L2.cuda()

        optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, generator2.parameters()), lr=opt.lr_g,
                                       betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
        # ------------------------------------------
        generator1.eval()
        generator2.train()
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
        )
        while True:
            for items in train_loader:
                generator2.iteration += 1

                images, images_gray, edges, masks = self.cuda(*items)

                # edge model
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                res1 = outputs_merged.detach()

                # kpn model
                img_rainy_kpn = images * (1-masks)
                kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                kpn_out = (kpn_out * masks) + (images * (1-masks))
                res2 = kpn_out.detach()
                uncertainty_map = core.detach()

                # kpn2
                input2 = torch.cat((res1, res2, uncertainty_map, masks), dim=1)
                input2_ = torch.cat((res1, res2), dim=1)
                kpn_out2, _ = generator2(input2, input2_)

                ssim_loss = -criterion_ssim(images, kpn_out2)
                Pixellevel_L1_Loss = criterion_L1(images, kpn_out2)

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
                    masks_ = torch.cat([masks] * 3, dim=1)
                    img_list = [img_rainy_kpn, masks_, kpn_out2, res1, res2, images]
                    name_list = ['in', 'mask', 'pred', 'edge_out', 'kpn_out', 'gt']
                    kpn_utils.save_sample_png(sample_folder=opt.train_sample, sample_name='ite_{}'.format(generator2.iteration),
                                          img_list=img_list, name_list=name_list, pixel_max_cnt=255)

                # eval
                if generator2.iteration % opt.eval_interval == 0:
                    generator2.eval()
                    cur_psnr, cur_ssim = self.fusion_eval(generator1, generator2, opt)
                    generator2.train()

                print('iteration:{}  ssim_loss:{}  l1_loss:{}  loss:{}'.format(generator2.iteration, ssim_loss.item(),
                                                                               Pixellevel_L1_Loss.item(), loss.item()))

    def fusion_eval(self, generator1, generator2, opt):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
        )

        psnr_list = []
        ssim_list = []
        for items in val_loader:
            images, images_gray, edges, masks = self.cuda(*items)
            with torch.no_grad():
                # edge model
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                res1 = outputs_merged.detach()

                # kpn model
                img_rainy_kpn = images * (1 - masks)
                kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                kpn_out = (kpn_out * masks) + (images * (1 - masks))
                res2 = kpn_out.detach()
                uncertainty_map = core.detach()

                # kpn2
                input2 = torch.cat((res1, res2, uncertainty_map, masks), dim=1)
                input2_ = torch.cat((res1, res2), dim=1)
                kpn_out2, _ = generator2(input2, input2_)
                kpn_out2_merged = (kpn_out2 * masks) + (images * (1 - masks))

                # sample
                if len(psnr_list) % opt.eval_sample_interval == 0:
                    masks_ = torch.cat([masks] * 3, dim=1)
                    img_list = [img_rainy_kpn, masks_, kpn_out2_merged, res1, res2,  images]
                    name_list = ['in', 'mask', 'pred', 'edge_out', 'kpn_out', 'gt']
                    kpn_utils.save_sample_png(sample_folder=opt.eval_sample,
                                              sample_name='ite_{}_{}'.format(generator2.iteration, len(psnr_list)),
                                              img_list=img_list, name_list=name_list, pixel_max_cnt=255)


                img_pred = kpn_utils.recover_process(kpn_out2_merged, height=-1, width=-1)
                img_gt = kpn_utils.recover_process(images, height=-1, width=-1)

                psnr = kpn_utils.psnr(img_pred, img_gt)
                ssim = compare_ssim(img_gt, img_pred, multichannel=True, data_range=255)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            print('iteration {}_{}  psnr:{}  ssim:{}'.format(generator2.iteration, len(psnr_list), psnr, ssim))

            if len(psnr_list) >= 1000:
                break

        ave_psnr = np.average(psnr_list)
        ave_ssim = np.average(ssim_list)
        print('iteration:{} psnr_ave:{} ssim_ave:{}'.format(generator2.iteration, ave_psnr, ave_ssim))

        return ave_psnr, ave_ssim


    def fusion_test(self):
        opt = get_opt()
        generator1 = kpn_utils_15.create_generator(opt)
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
        # ------------------------------------------

        generator1.eval()
        generator2.eval()
        self.edge_model.eval()
        self.inpaint_model.eval()

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )



        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []

        kpn_psnr_list = []
        kpn_ssim_list = []

        binary_psnr_list = []
        binary_ssim_list = []

        edge_psnr_list = []
        edge_ssim_list = []
        edge_l1_list = []
        edge_lpips_list = []

        input_psnr_list = []
        input_ssim_list = []



        kpn1_time = []
        gan_time = []
        smart_time = []
        binary_time = []
        with torch.no_grad():
            for i in range(1):
                for items in test_loader:
                    images, images_gray, edges, masks = self.cuda(*items)

                    # edge model
                    t = time.time()
                    edges = self.edge_model(images_gray, edges, masks).detach()

                    outputs = self.inpaint_model(images, edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))
                    gan_time.append(time.time() - t)

                    res1 = outputs_merged.detach()

                    edge_psnr, edge_ssim = npk_val(res1, images)
                    edge_psnr_list.append(edge_psnr)
                    edge_ssim_list.append(edge_ssim)

                    edge_l1_loss = torch.nn.functional.l1_loss(res1, images, reduction='mean').item()
                    edge_l1_list.append(edge_l1_loss)

                    #0.09068728238344193
                    edge_pl = loss_fn_vgg(transf(res1[0].cpu()).cuda(), transf(images[0].cpu()).cuda()).item()
                    edge_lpips_list.append(edge_pl)



                    # kpn model
                    img_rainy_kpn = images * (1 - masks)
                    t = time.time()
                    kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                    kpn_out_merged = (kpn_out * masks) + (images * (1 - masks))
                    kpn1_time.append(time.time()-t)

                    res2 = kpn_out_merged.detach()
                    uncertainty_map = core.detach()

                    # smart fusion
                    input2 = torch.cat((res1, res2, uncertainty_map, masks), dim=1)
                    input2_ = torch.cat((res1, res2), dim=1)
                    t = time.time()
                    kpn_out2, _ = generator2(input2, input2_)
                    kpn_out2_merged = (kpn_out2 * masks) + (images * (1 - masks))
                    smart_time.append(time.time() - t)

                    l1_loss = torch.nn.functional.l1_loss(kpn_out2_merged, images, reduction='mean').item()
                    l1_list.append(l1_loss)

                    pl = loss_fn_vgg(transf(kpn_out2_merged[0].cpu()).cuda(), transf(images[0].cpu()).cuda()).item()
                    lpips_list.append(pl)


                    img_pred = kpn_utils.recover_process(kpn_out2_merged, height=-1, width=-1)
                    img_gt = kpn_utils.recover_process(images, height=-1, width=-1)

                    psnr = kpn_utils.psnr(img_pred, img_gt)
                    ssim = compare_ssim(img_gt, img_pred, multichannel=True, data_range=255)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)


                    # binary fusion--------------------

                    uncertainty = get_uncertainty(core, masks)
                    uncertainty = torch.from_numpy(uncertainty).float()
                    if torch.cuda.is_available():
                        uncertainty = uncertainty.cuda()

                    t = time.time()
                    binary_pre = res2 * (1 - uncertainty) + res1 * uncertainty
                    binary_time.append(time.time() - t)
                    binary_psnr, binary_ssim = npk_val(binary_pre, images)

                    binary_psnr_list.append(binary_psnr)
                    binary_ssim_list.append(binary_ssim)


                    # kpn--------------------------------
                    kpn_psnr, kpn_ssim = npk_val(res2, images)
                    kpn_psnr_list.append(kpn_psnr)
                    kpn_ssim_list.append(kpn_ssim)

                    #input gt----------------------------
                    input_psnr, input_ssim = npk_val(img_rainy_kpn, images)
                    input_psnr_list.append(input_psnr)
                    input_ssim_list.append(input_ssim)


                    # sample
                    if len(psnr_list) % 1 == 0:
                        # different-------------------------------------
                        #get_different(kpn_out, outputs, kpn_out2, images, opt.test_sample, len(psnr_list))


                        # masks_ = torch.cat([masks] * 3, dim=1)
                        # un_map = uncertainty.unsqueeze(0).unsqueeze(0)
                        # un_map = torch.cat([un_map] * 3, dim=1)
                        # img_list = [img_rainy_kpn, masks_, kpn_out2, res1, res2, images, un_map]
                        # name_list = ['in', 'mask', 'pred', 'edge_out', 'kpn_out', 'gt', 'un_map']
                        # kpn_utils.save_sample_png(sample_folder=opt.test_sample,
                        #                           sample_name='{}'.format(len(psnr_list)),
                        #                           img_list=img_list, name_list=name_list, pixel_max_cnt=255)

                        masks_ = torch.cat([masks] * 3, dim=1)
                        un_map = uncertainty.unsqueeze(0).unsqueeze(0)
                        un_map = torch.cat([un_map] * 3, dim=1)
                        img_list = [kpn_out2_merged, res1, images]
                        name_list = ['pred', 'edge_out', 'gt']
                        kpn_utils.save_sample_png(sample_folder=opt.test_sample,
                                                  sample_name='{}'.format(len(psnr_list)),
                                                  img_list=img_list, name_list=name_list, pixel_max_cnt=255)


                    print('{}  psnr:{}/{}  ssim:{}/{}  kpn_psnr:{}/{}  kpn_ssim:{}/{}  '
                          'binary_psnr:{}/{}  binary_ssim:{}/{}  edge_psnr:{}/{}  edge_ssim:{}/{}'
                          '  input_psnr:{}/{}  input_ssim:{}/{}'
                          ' edge_l1:{}/{}  edge_lpips:{}/{}'
                          ' final_l1:{}/{} final_lpips:{}/{}'.format(
                        len(psnr_list),
                        psnr, np.average(psnr_list), ssim, np.average(ssim_list),
                        kpn_psnr, np.average(kpn_psnr_list), kpn_ssim, np.average(kpn_ssim_list),
                        binary_psnr, np.average(binary_psnr_list), binary_ssim, np.average(binary_ssim_list),
                        edge_psnr, np.average(edge_psnr_list), edge_ssim, np.average(edge_ssim_list),
                        input_psnr, np.average(input_psnr_list), input_ssim, np.average(input_ssim_list),
                        edge_pl, np.average(edge_l1_list), edge_pl, np.average(edge_lpips_list),
                        l1_loss, np.average(l1_list), pl, np.average(lpips_list))
                    )

                    print('--------time    {} kpn1:{}  binary:{}  smart:{}  gan:{}'.format(len(smart_time), np.average(kpn1_time),
                                                                np.average(binary_time),
                                                                np.average(smart_time),
                                                                np.average(gan_time), ))
                    if len(smart_time) >= 20000:
                        break
        print('final psnr:{}  ssim:{} l1:{} lpips:{}  kpn_psnr:{}  kpn_ssim:{} '
              'binary_psnr:{}  binary_ssim:{} edge_psnr:{}  edge_ssim:{} edge_l1:{}  edge_lpips:{}'
              ' input_psnr:{}  input_ssim:{} '.format(
            np.average(psnr_list), np.average(ssim_list), np.average(l1_list), np.average(lpips_list),
            np.average(kpn_psnr_list), np.average(kpn_ssim_list),
            np.average(binary_psnr_list), np.average(binary_ssim_list),
            np.average(edge_psnr_list), np.average(edge_ssim_list), np.average(edge_l1_list), np.average(edge_lpips_list),
            np.average(input_psnr_list), np.average(input_ssim_list))
        )

        print('kpn1:{}  binary:{}  smart:{}  gan:{}'.format(np.average(kpn1_time),
                                                            np.average(binary_time),
                                                            np.average(smart_time),
                                                            np.average(gan_time), ))


    def adjust_learning_rate(self, opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

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

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

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
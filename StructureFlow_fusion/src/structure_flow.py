import math
import os
import shutil
from itertools import islice

import numpy as np
import tensorboardX
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.measure import compare_ssim
# from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader

import kpn.pytorch_ssim as pytorch_ssim
import kpn.utils as kpn_utils
from kpn.config import get_opt
from .data import Dataset
from .models import StructureFlowModel
from .utils import Progbar, write_2images, write_2tensorboard
from kpn.validation import npk_val, get_uncertainty, get_uncertainty_2
from different import get_different
import time
import torchvision
import lpips


class StructureFlow():
    def __init__(self, config):
        self.config = config
        self.debug=False
        self.flow_model = StructureFlowModel(config).to(config.DEVICE)

        self.samples_path_train = os.path.join(config.PATH, config.NAME, 'images_train')
        self.samples_path_eval = os.path.join(config.PATH, config.NAME, 'images_eval')
        self.samples_path_test = os.path.join(config.PATH, config.NAME, 'images_test')
        self.checkpoints_path = os.path.join(config.PATH, config.NAME, 'checkpoints')
        self.test_image_path = os.path.join(config.PATH, config.NAME, 'test_result')

        if self.config.MODE == 'train' and not self.config.RESUME_ALL:
            pass
        else:
            self.flow_model.load(self.config.WHICH_ITER)

        if self.config.MODEL == 1:
            self.stage_name='structure_reconstructor'
        elif self.config.MODEL == 2:
            self.stage_name='texture_generator'
        elif self.config.MODEL == 3:
            self.stage_name='joint_train'

    def train(self):
        train_writer = self.obtain_log(self.config)
        train_dataset = Dataset(self.config.DATA_TRAIN_GT, self.config.DATA_TRAIN_STRUCTURE, 
                                self.config, self.config.DATA_MASK_FILE)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config.TRAIN_BATCH_SIZE, 
                                  shuffle=True, drop_last=True, num_workers=0)

        val_dataset = Dataset(self.config.DATA_VAL_GT, self.config.DATA_VAL_STRUCTURE, 
                              self.config, self.config.DATA_MASK_FILE)
        sample_iterator = val_dataset.create_iterator(self.config.SAMPLE_SIZE)


        iterations = self.flow_model.iterations  
        total = len(train_dataset) 
        epoch = math.floor(iterations*self.config.TRAIN_BATCH_SIZE/total)
        keep_training = True
        model = self.config.MODEL
        max_iterations = int(float(self.config.MAX_ITERS))

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            # progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                inputs, smooths, gts, maps = self.cuda(*items)

                # structure model
                if model == 1:
                    logs = self.flow_model.update_structure(inputs, smooths, maps)
                    iterations = self.flow_model.iterations
                # flow model
                elif model == 2:
                    logs = self.flow_model.update_flow(inputs, smooths, gts, maps, self.flow_model.use_correction_loss, self.flow_model.use_vgg_loss)
                    iterations = self.flow_model.iterations
                # flow with structure model
                elif model == 3:
                    with torch.no_grad(): 
                        smooth_stage_1 = self.flow_model.structure_forward(inputs, smooths, maps)
                    logs = self.flow_model.update_flow(inputs, smooth_stage_1.detach(), gts, maps, self.flow_model.use_correction_loss, self.flow_model.use_vgg_loss)
                    iterations = self.flow_model.iterations

                if iterations >= max_iterations:
                    keep_training = False
                    break

                # print(logs)
                logs = [
                    ("epoch", epoch),
                    ("iter", iterations),
                ] + logs

                # progbar.add(len(inputs), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                print('train_{}'.format(logs))

                # log model 
                if self.config.LOG_INTERVAL and iterations % self.config.LOG_INTERVAL == 0:
                    self.write_loss(logs, train_writer)
                # sample model 
                if self.config.SAMPLE_INTERVAL and iterations % self.config.SAMPLE_INTERVAL == 0:
                    items = next(sample_iterator)
                    inputs, smooths, gts, maps = self.cuda(*items)
                    result,flow = self.flow_model.sample(inputs, smooths, gts, maps)
                    self.write_image(result, train_writer, iterations, 'image')
                    self.write_image(flow,   train_writer, iterations, 'flow')
                # evaluate model 
                if self.config.EVAL_INTERVAL and iterations % self.config.EVAL_INTERVAL == 0:
                    self.flow_model.eval()
                    print('\nstart eval...\n')
                    self.eval(writer=train_writer)
                    self.flow_model.train()

                # save the latest model 
                if self.config.SAVE_LATEST and iterations % self.config.SAVE_LATEST == 0:
                    print('\nsaving the latest model (total_steps %d)\n' % (iterations))
                    self.flow_model.save('latest')

                # save the model 
                if self.config.SAVE_INTERVAL and iterations % self.config.SAVE_INTERVAL == 0:
                    print('\nsaving the model of iterations %d\n' % iterations)
                    self.flow_model.save(iterations)
        print('\nEnd training....')


    def eval(self, writer=None):
        val_dataset = Dataset(self.config.DATA_VAL_GT , self.config.DATA_VAL_STRUCTURE, self.config, self.config.DATA_VAL_MASK)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size = 1,
            shuffle=False
        )
        model = self.config.MODEL
        total = len(val_dataset)
        iterations = self.flow_model.iterations

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        psnr_list = []
        ssim_list = []

        train_writer = self.obtain_log(self.config)

        # TODO: add fid score to evaluate
        with torch.no_grad(): 
            # for items in val_loader:
            for j, items in enumerate(islice(val_loader, len(val_loader))):

                logs = []
                iteration += 1
                inputs, smooths, gts, maps = self.cuda(*items)
                if model == 1:
                    outputs_structure = self.flow_model.structure_forward(inputs, smooths, maps)
                    psnr, ssim, l1 = self.metrics(outputs_structure, smooths)
                    logs.append(('psnr', psnr.item()))
                    psnr_list.append(psnr.item())


                    result = [inputs, gts, maps, outputs_structure]
                    # self.write_image(result, train_writer, j, 'image')


                # inpaint model
                elif model == 2:
                    outputs, flow_maps = self.flow_model.flow_forward(inputs, smooths, maps)
                    psnr, ssim, l1 = self.metrics(outputs, gts)
                    logs.append(('psnr', psnr.item()))
                    logs.append(('ssim', ssim.item()))
                    psnr_list.append(psnr.item())
                    ssim_list.append(ssim.item())

                    result = [inputs, gts, maps, outputs]
                    # self.write_image(result, train_writer, j, 'image')


                # inpaint with structure model
                elif model == 3:
                    smooth_stage_1 = self.flow_model.structure_forward(inputs, smooths, maps)
                    outputs, flow_maps = self.flow_model.flow_forward(inputs, smooth_stage_1, maps)
                    psnr, ssim, l1 = self.metrics(outputs, gts)
                    logs.append(('psnr', psnr.item()))
                    logs.append(('ssim', ssim.item()))
                    psnr_list.append(psnr.item())
                    ssim_list.append(ssim.item())

                    result = [inputs, gts, maps, outputs]
                    # self.write_image(result, train_writer, j, 'image')

                logs = [("it", iteration), ] + logs
                # progbar.add(len(inputs), values=logs)
                print('eval_{}'.format(logs))

        avg_psnr = np.average(psnr_list)
        avg_ssim = np.average(ssim_list)

        if writer is not None:
            writer.add_scalar('eval_psnr', avg_psnr, iterations)

        print('model eval at iterations:%d'%iterations)
        print('average psnr:{}  ssim:{}'.format(avg_psnr, avg_ssim))


    def test(self):
        self.flow_model.eval()

        train_writer = self.obtain_log(self.config)
        test_dataset = Dataset(self.config.DATA_TEST_GT, self.config.DATA_TEST_STRUCTURE, self.config,
                              self.config.DATA_TEST_MASK)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False
        )
        model = self.config.MODEL
        total = len(test_dataset)
        iterations = self.flow_model.iterations

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        psnr_list = []
        ssim_list = []


        with torch.no_grad():
            # for items in val_loader:
            for j, items in enumerate(islice(test_loader, len(test_loader))):

                logs = []
                iteration += 1
                inputs, smooths, gts, maps = self.cuda(*items)
                if model == 1:
                    outputs_structure = self.flow_model.structure_forward(inputs, smooths, maps)
                    psnr, ssim, l1 = self.metrics(outputs_structure, smooths)
                    logs.append(('psnr', psnr.item()))
                    psnr_list.append(psnr.item())

                # inpaint model
                elif model == 2:
                    outputs, flow_maps = self.flow_model.flow_forward(inputs, smooths, maps)
                    psnr, ssim, l1 = self.metrics(outputs, gts)
                    logs.append(('psnr', psnr.item()))
                    logs.append(('ssim', ssim.item()))
                    psnr_list.append(psnr.item())
                    ssim_list.append(ssim.item())

                # inpaint with structure model
                elif model == 3:
                    smooth_stage_1 = self.flow_model.structure_forward(inputs, smooths, maps)
                    outputs, flow_maps = self.flow_model.flow_forward(inputs, smooth_stage_1, maps)

                    outputs = outputs*maps + gts*(1-maps)

                    psnr, ssim = self.metrics_2(outputs, gts)
                    logs.append(('psnr', psnr))
                    logs.append(('ssim', ssim))
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    print('psnr:{}/{}   ssim:{}/{} {}'.format(psnr, np.average(psnr_list), ssim, np.average(ssim_list), len(ssim_list)))

                    #sample
                    # if len(psnr_list) % 1 == 0:
                    #     result = [inputs, smooths, gts, maps, smooth_stage_1, outputs]
                    #     self.write_image(result, train_writer, len(ssim_list), 'image')

                    if len(ssim_list) % 100 == 0:
                        masks_ = torch.cat([maps] * 3, dim=1)
                        img_list = [self.postprocess(inputs), masks_, self.postprocess(outputs), self.postprocess(gts)]

                        name_list = ['in', 'mask', 'pred', 'gt']
                        kpn_utils.save_sample_png(sample_folder='./result/test_20_structure',
                                                  sample_name='ite_{}_{}'.format(0, len(psnr_list)),
                                                  img_list=img_list, name_list=name_list, pixel_max_cnt=255)

        avg_psnr = np.average(psnr_list)
        avg_ssim = np.average(ssim_list)

        print('model eval at iterations:%d' % iterations)
        print('average psnr:{}  ssim:{}'.format(avg_psnr, avg_ssim))


    def fusion_train(self):
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
        self.flow_model.eval()

        train_writer = self.obtain_log(self.config)
        train_dataset = Dataset(self.config.DATA_TRAIN_GT, self.config.DATA_TRAIN_STRUCTURE, self.config,
                               self.config.DATA_MASK_FILE)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.train_batch_size,
            shuffle=True
        )

        while True:
            for items in train_loader:

                inputs, smooths, gts, maps = self.cuda(*items)
                generator2.iteration += 1

                with torch.no_grad():
                    smooth_stage_1 = self.flow_model.structure_forward(inputs, smooths, maps)
                    outputs, flow_maps = self.flow_model.flow_forward(inputs, smooth_stage_1, maps)
                    outputs = outputs * maps + gts * (1 - maps)
                    res1 = outputs.detach()

                    # kpn model
                    img_rainy_kpn = self.postprocess(gts) * (1 - maps)
                    kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                    kpn_out = kpn_out * maps + self.postprocess(gts) * (1-maps)
                    res2 = kpn_out.detach()
                    uncertainty_map = core.detach()

                # kpn2
                uMap = get_uncertainty(uncertainty_map, gts.size(0))
                input2 = torch.cat((res1, res2, uncertainty_map, uMap), dim=1)
                input2_ = torch.cat((res1, res2), dim=1)
                kpn_out2, _ = generator2(input2, input2_)

                ssim_loss = -criterion_ssim(gts, kpn_out2)
                Pixellevel_L1_Loss = criterion_L1(gts, kpn_out2)

                loss = Pixellevel_L1_Loss + 0.2 * ssim_loss

                optimizer_G.zero_grad()

                loss.backward()
                optimizer_G.step()

                # save
                if generator2.iteration % opt.save_by_iter == 0:
                    self.save_model(opt, generator2.iteration, generator2)

                # sample
                if generator2.iteration % opt.train_sample_interval == 0:
                    result = [inputs, smooth_stage_1, res1, res2, kpn_out2, gts]
                    self.write_image(result, train_writer, generator2.iteration, 'image_train', self.samples_path_train)


                #eval
                if generator2.iteration % opt.eval_interval == 0:
                    generator2.eval()
                    self.fusion_eval(generator1, generator2, opt, train_writer)
                    generator2.train()

                print('ite:{} l1:{}  ssim:{}  loss:{}'.format(generator2.iteration, Pixellevel_L1_Loss.item(), ssim_loss.item(), loss.item()))


    def fusion_eval(self, generator1, generator2, opt, train_writer):
        valid_dataset = Dataset(self.config.DATA_VAL_GT, self.config.DATA_VAL_STRUCTURE, self.config,
                                self.config.DATA_VAL_MASK)
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False
        )
        psnr_list = []
        ssim_list = []
        for items in valid_loader:
            inputs, smooths, gts, maps = self.cuda(*items)

            with torch.no_grad():
                smooth_stage_1 = self.flow_model.structure_forward(inputs, smooths, maps)
                outputs, flow_maps = self.flow_model.flow_forward(inputs, smooth_stage_1, maps)
                outputs = outputs * maps + gts * (1 - maps)
                res1 = outputs.detach()

                # kpn model
                img_rainy_kpn = self.postprocess(gts) * (1 - maps)
                kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                kpn_out = kpn_out * maps + self.postprocess(gts) * (1 - maps)
                res2 = kpn_out.detach()
                uncertainty_map = core.detach()

                # kpn2
                uMap = get_uncertainty(uncertainty_map, gts.size(0))
                input2 = torch.cat((res1, res2, uncertainty_map, uMap), dim=1)
                input2_ = torch.cat((res1, res2), dim=1)
                kpn_out2, _ = generator2(input2, input2_)
                kpn_out2 = kpn_out2*maps + (1-maps)*gts

            psnr, ssim, l1 = self.metrics(kpn_out2, gts)
            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())
            print('psnr:{}/{}  ssim:{}/{}  {}'.format(psnr.item(), np.average(psnr_list), ssim.item(), np.average(ssim_list), len(ssim_list)))

            # sample
            if len(ssim_list) % opt.eval_sample_interval == 0:
                result = [inputs, smooth_stage_1, res1, res2, kpn_out2, gts]
                self.write_image(result, train_writer, generator2.iteration, 'image_{}'.format(len(ssim_list)), self.samples_path_eval)


            if len(ssim_list) >= 1000:
                break

        print('psnr_ave:{}  ssim_ave:'.format(np.average(psnr_list), np.average(ssim_list)))

    def fusion_test(self):
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

        # ------------------------------------------
        generator1.eval()
        generator2.eval()
        self.flow_model.eval()

        train_writer = self.obtain_log(self.config)
        test_dataset = Dataset(self.config.DATA_TEST_GT, self.config.DATA_TEST_STRUCTURE, self.config,
                                self.config.DATA_TEST_MASK)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False
        )
        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []

        kpn_psnr_list = []
        kpn_ssim_list = []

        binary_psnr_list = []
        binary_ssim_list = []

        str_psnr_list = []
        str_ssim_list = []
        str_l1_list = []
        str_lpips_list = []

        input_psnr_list = []
        input_ssim_list = []

        kpn1_time = []
        gan_time = []
        smart_time = []
        binary_time = []

        for i in range(1):
            for items in test_loader:
                inputs, smooths, gts, maps = self.cuda(*items)

                with torch.no_grad():
                    t = time.time()
                    smooth_stage_1 = self.flow_model.structure_forward(inputs, smooths, maps)
                    outputs, flow_maps = self.flow_model.flow_forward(inputs, smooth_stage_1, maps)
                    outputs_merged = outputs * maps + gts * (1 - maps)
                    gan_time.append(time.time() - t)
                    res1 = outputs_merged.detach()

                    # kpn model
                    img_rainy_kpn = self.postprocess(gts) * (1 - maps)
                    gt_recover = self.postprocess(gts)
                    t = time.time()
                    kpn_out, core = generator1(img_rainy_kpn, img_rainy_kpn)
                    kpn_out_merged = kpn_out * maps + gt_recover * (1 - maps)
                    kpn1_time.append(time.time() - t)

                    res2 = kpn_out_merged.detach()
                    uncertainty_map = core.detach()

                    # kpn2
                    uMap = get_uncertainty(uncertainty_map, gts.size(0))
                    input2 = torch.cat((res1, res2, uncertainty_map, uMap), dim=1)
                    input2_ = torch.cat((res1, res2), dim=1)
                    t = time.time()
                    kpn_out2, _ = generator2(input2, input2_)
                    kpn_out2 = kpn_out2 * maps + (1 - maps) * gts
                    smart_time.append(time.time() - t)

                psnr, ssim = self.metrics_2(kpn_out2, gts)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                l1_loss = torch.nn.functional.l1_loss(self.postprocess(kpn_out2), self.postprocess(gts), reduction='mean').item()
                l1_list.append(l1_loss)

                pl = loss_fn_vgg(transf(self.postprocess(kpn_out2)[0].cpu()).cuda(), transf(self.postprocess(gts)[0].cpu()).cuda()).item()
                lpips_list.append(pl)


                # binary fusion--------------------

                uncertainty = get_uncertainty_2(core, gts.size(0))
                if torch.cuda.is_available():
                    uncertainty = uncertainty.cuda()

                t = time.time()
                binary_pre = res2 * (1 - uncertainty) + self.postprocess(res1) * uncertainty
                binary_time.append(time.time() - t)

                binary_psnr, binary_ssim = npk_val(binary_pre, self.postprocess(gts))

                binary_psnr_list.append(binary_psnr)
                binary_ssim_list.append(binary_ssim)

                # kpn--------------------------------
                kpn_psnr, kpn_ssim = npk_val(res2, self.postprocess(gts))
                kpn_psnr_list.append(kpn_psnr)
                kpn_ssim_list.append(kpn_ssim)

                # str--------------------------------
                str_psnr, str_ssim = self.metrics_2(res1, gts)
                str_psnr_list.append(str_psnr)
                str_ssim_list.append(str_ssim)

                str_l1_loss = torch.nn.functional.l1_loss(self.postprocess(res1), self.postprocess(gts), reduction='mean').item()
                str_l1_list.append(str_l1_loss)

                str_pl = loss_fn_vgg(transf(self.postprocess(res1)[0].cpu()).cuda(), transf(self.postprocess(gts)[0].cpu()).cuda()).item()
                str_lpips_list.append(str_pl)



                # input gt -----------------------------------
                input_psnr, input_ssim = npk_val(img_rainy_kpn, self.postprocess(gts))
                input_psnr_list.append(input_psnr)
                input_ssim_list.append(input_ssim)

                # sample
                # if len(ssim_list) % opt.test_interval == 0:
                #     result = [inputs, smooth_stage_1, res1, res2, kpn_out2, gts]
                #     self.write_image(result, train_writer, generator2.iteration, 'image_{}'.format(len(ssim_list)), self.samples_path_test)

                if len(ssim_list) % 1 == 0:
                    # different-------------------------------------
                    # get_different(kpn_out, outputs, kpn_out2, gts, opt.test_sample, len(psnr_list))
                    #
                    # masks_ = torch.cat([maps] * 3, dim=1)
                    # img_list = [img_rainy_kpn, masks_, self.postprocess(kpn_out2),
                    #             self.postprocess(res1), res2, self.postprocess(gts)]
                    #
                    # name_list = ['in', 'mask', 'pred', 'edge_out', 'kpn_out', 'gt']
                    # kpn_utils.save_sample_png(sample_folder=opt.test_sample,
                    #                           sample_name='{}_'.format(len(psnr_list)),
                    #                           img_list=img_list, name_list=name_list, pixel_max_cnt=255)

                    masks_ = torch.cat([maps] * 3, dim=1)
                    img_list = [self.postprocess(kpn_out2),
                                self.postprocess(res1), self.postprocess(gts)]

                    name_list = ['pred', 'edge_out',  'gt']
                    kpn_utils.save_sample_png(sample_folder=opt.test_sample,
                                              sample_name='{}_'.format(len(psnr_list)),
                                              img_list=img_list, name_list=name_list, pixel_max_cnt=255)

                print(
                    '{}  psnr:{}/{}  ssim:{}/{} l1:{}/{} lpips:{}/{}  kpn_psnr:{}/{}  kpn_ssim:{}/{}  '
                    'binary_psnr:{}/{}  binary_ssim:{}/{}   str_psnr:{}/{}  str_ssim:{}/{} str_l1:{}/{} str_lpips:{}/{}'
                    'input_psnr:{}/{}  input_ssim:{}/{}'.format(
                        len(psnr_list),
                        psnr, np.average(psnr_list), ssim, np.average(ssim_list), l1_loss, np.average(l1_list), pl, np.average(lpips_list),
                        kpn_psnr, np.average(kpn_psnr_list), kpn_ssim, np.average(kpn_ssim_list),
                        binary_psnr, np.average(binary_psnr_list), binary_ssim, np.average(binary_ssim_list),
                        str_psnr, np.average(str_psnr_list), str_ssim, np.average(str_ssim_list), str_l1_loss, np.average(str_l1_list), str_pl, np.average(str_lpips_list),
                        input_psnr, np.average(input_psnr_list), input_ssim, np.average(input_ssim_list))
                )

                print('{}  kpn1:{}  binary:{}  smart:{}  gan:{}'.format(len(ssim_list), np.average(kpn1_time),
                                                                np.average(binary_time),
                                                                np.average(smart_time),
                                                                np.average(gan_time), ))
                if len(smart_time) >= 20000:
                    break
        print(
            'final psnr:{}  ssim:{} l1:{} lpips:{}  kpn_psnr:{}  kpn_ssim:{}  binary_psnr:{}  binary_ssim:{}   str_psnr:{}  str_ssim:{} str_l1:{} str_lpips:{}'
            '  input_psnr:{}  input_ssim:{}'.format(
                np.average(psnr_list), np.average(ssim_list), np.average(l1_list), np.average(lpips_list),
                np.average(kpn_psnr_list), np.average(kpn_ssim_list),
                np.average(binary_psnr_list), np.average(binary_ssim_list),
                np.average(str_psnr_list), np.average(str_ssim_list), np.average(str_l1_list), np.average(str_lpips_list),
                np.average(input_psnr_list), np.average(input_ssim_list))
        )

        print('kpn1:{}  binary:{}  smart:{}  gan:{}'.format(np.average(kpn1_time),
                                                            np.average(binary_time),
                                                            np.average(smart_time),
                                                            np.average(gan_time), ))


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

    def obtain_log(self, config):
        log_dir = os.path.join(config.PATH, config.NAME, self.stage_name+'_log')
        if os.path.exists(log_dir) and config.REMOVE_LOG:
            shutil.rmtree(log_dir)
        train_writer = tensorboardX.SummaryWriter(log_dir)
        return train_writer


    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)


    def write_loss(self, logs, train_writer):
        iteration = [x[1] for x in logs if x[0]=='iter']
        for x in logs:
            if x[0].startswith('l_'):
                train_writer.add_scalar(x[0], x[1], iteration[-1])

    def write_image(self, result, train_writer, iterations, label, p):
        if result:
            name = '%s/model%d_sample_%08d'%(p, self.config.MODEL, iterations) + label + '.jpg'
            write_2images(result, self.config.SAMPLE_SIZE, name)
            write_2tensorboard(iterations, result, train_writer, self.config.SAMPLE_SIZE, label)


    def postprocess(self, x):
        x = (x + 1) / 2
        x.clamp_(0, 1) 
        return x   

    def metrics(self, inputs, gts):
        inputs = self.postprocess(inputs)
        gts = self.postprocess(gts)
        psnr_value=[]
        l1_value = torch.mean(torch.abs(inputs-gts))

        [b,n,w,h] = inputs.size()
        inputs = (inputs*255.0).int().float()/255.0
        gts    = (gts*255.0).int().float()/255.0

        for i in range(inputs.size(0)):
            inputs_p = inputs[i,:,:,:].cpu().numpy().astype(np.float32).transpose(1,2,0)
            gts_p = gts[i,:,:,:].cpu().numpy().astype(np.float32).transpose(1,2,0)
            psnr_value.append(compare_psnr(inputs_p, gts_p, data_range=1))

        psnr_value = np.average(psnr_value)            
        inputs = inputs.view(b*n, w, h).cpu().numpy().astype(np.float32).transpose(1,2,0)
        gts = gts.view(b*n, w, h).cpu().numpy().astype(np.float32).transpose(1,2,0)
        ssim_value = compare_ssim(inputs, gts, data_range=1, win_size=51, multichannel=True)
        return psnr_value, ssim_value, l1_value


    def metrics_2(self, pre, gts):
        pre = self.postprocess(pre)
        gts = self.postprocess(gts)

        img_pred = kpn_utils.recover_process(pre, height=-1, width=-1)
        img_gt = kpn_utils.recover_process(gts, height=-1, width=-1)

        psnr = kpn_utils.psnr(img_pred, img_gt)
        ssim = compare_ssim(img_gt, img_pred, multichannel=True, data_range=255)

        return psnr, ssim




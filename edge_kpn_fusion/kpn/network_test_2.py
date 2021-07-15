import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from numpy import *
# import utils

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------


def weights_init(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#        Self-Guided Network (SGN)
# ----------------------------------------
# KPN基本网路单元


class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1,
                          kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(
                fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(
                fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm


class KPN(nn.Module):
    def __init__(self, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * \
                    (burst_length if blind_est else burst_length + 1)
        out_channel = (3 if color else 1) * (
            2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512 + 512, 512, channel_att=channel_att,
                           spatial_att=spatial_att)
        self.conv7 = Basic(256 + 512, 256, channel_att=channel_att,
                           spatial_att=spatial_att)

        #从此处开始分支
        self.branch_conv_1 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.branch_conv_2 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.branch_conv_3 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.branch_conv_4 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)

        self.outc_1 = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        self.outc_2 = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        self.outc_3 = nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        self.outc_4 = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred_1 = KernelConv(kernel_size, sep_conv, self.core_bias)
        self.kernel_pred_2 = KernelConv(kernel_size, sep_conv, self.core_bias)
        self.kernel_pred_3 = KernelConv(kernel_size, sep_conv, self.core_bias)
        self.kernel_pred_4 = KernelConv(kernel_size, sep_conv, self.core_bias)


    # 前向传播函数
    def forward(self, data_with_est, data, eps, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(
            conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(
            conv6, scale_factor=2, mode=self.upMode)], dim=1))
        # conv8 = self.conv8(torch.cat([conv2, F.interpolate(
        #     conv7, scale_factor=2, mode=self.upMode)], dim=1))
        # # return channel K*K*N
        # core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))

                # return channel K*K*N
        
        branch_conv_1 = self.branch_conv_1(torch.cat(
            [conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        branch_conv_2 = self.branch_conv_2(torch.cat(
            [conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        branch_conv_3 = self.branch_conv_3(torch.cat(
            [conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        branch_conv_4 = self.branch_conv_4(torch.cat(
            [conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))

        #four heads
        core_1 = self.outc_1(F.interpolate(
            branch_conv_1, scale_factor=2, mode=self.upMode))
        core_2 = self.outc_2(F.interpolate(
            branch_conv_2, scale_factor=2, mode=self.upMode))
        core_3 = self.outc_3(F.interpolate(
            branch_conv_3, scale_factor=2, mode=self.upMode))
        core_4 = self.outc_4(F.interpolate(
            branch_conv_4, scale_factor=2, mode=self.upMode))

        output_1 = self.kernel_pred_1(data, core_1, white_level)
        output_2 = self.kernel_pred_2(data, core_2, white_level)
        output_3 = self.kernel_pred_3(data, core_3, white_level)
        output_4 = self.kernel_pred_4(data, core_4, white_level)
        
        return output_1,output_2,output_3,output_4


class KernelConv(nn.Module):
    """
    the class of computing prediction
    """

    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur +
                        K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur +
                        K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum(
                'ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:,
                                             :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        # print(len(frames.size()))
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(
                core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(
                core, batch_size, N, color, height, width)

        # kernel = self.kernel_size[::-1]
        # uncertainty_map=torch.ones((batch_size,core[kernel[0]].shape[4],core[kernel[0]].shape[5]))
        # for i in range(batch_size):
        #     uncertainty_map[i]=torch.max(core[kernel[0]][i].squeeze()[0].squeeze(0),dim=0)[0]
        # uncertainty_map=uncertainty_map.cuda()

        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
                for i in range(K):
                    for j in range(K):
                        img_stack.append(
                            frame_pad[..., i:i + height, j:j + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        # print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False).squeeze()
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        # print('white_level', white_level.size())
        pred_img_i = pred_img_i / white_level
        #pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        return pred_img_i


class LossFunc(nn.Module):
    """
    loss function of KPN
    """

    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)


class LossBasic(nn.Module):
    """
    Basic loss function.
    """

    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) + \
            self.l1_loss(self.gradient(pred), self.gradient(ground_truth))


class LossAnneal(nn.Module):
    """
    anneal loss function
    """

    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """

    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) +
                torch.pow((u - d)[..., 0:w, 0:h], 2)
            )


def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net


class Atten_model(nn.Module):
    def __init__(self, kpn):
        super(Atten_model, self).__init__()

        self.conv1 = kpn.conv1
        self.conv2 = kpn.conv2
        self.conv3 = kpn.conv3
        self.conv4 = kpn.conv4
        self.conv5 = kpn.conv5
        self.conv6 = kpn.conv6
        self.conv7 = kpn.conv7

        self.branch_conv_light = kpn.branch_conv_3
        self.branch_conv_heavy = kpn.branch_conv_4
        self.outc_light = kpn.outc_3
        self.outc_heavy = kpn.outc_4

        self.kernel_pred_light = kpn.kernel_pred_3
        self.kernel_pred_heavy = kpn.kernel_pred_4

        for p in self.parameters():
            p.requires_grad = False
        
        #KPN2
        # 2~5层都是均值池化+3层卷积
        self.KPN2_conv1 = Basic(489, 64, channel_att=False, spatial_att=False)
        self.KPN2_conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.KPN2_conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.KPN2_conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.KPN2_conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.KPN2_conv6 = Basic(512 + 512, 512, channel_att=False,spatial_att=False)
        self.KPN2_conv7 = Basic(256 + 512, 256, channel_att=False,spatial_att=False)

        #从此处开始分支
        self.KPN2_conv8 = Basic(256+128, 3*3*6, channel_att=False, spatial_att=False)
        self.KPN2_outc = nn.Conv2d(3*3*6, 3*3*6, 1, 1, 0)
        self.KPN2_kernel_pred = KernelConv([3], False, False)
        self.KPN2_conv9 = nn.Conv2d(6, 3, 3, 1, 1)


       
       
    def forward(self, data_with_est, data, white_level=1.0):
        """
                forward and obtain pred image directly
                :param data_with_est: if not blind estimation, it is same as data
                :param data:
                :return: pred_img_i and img_pred
                """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection, 2,3,4对应8,7,6
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(
            conv5, scale_factor=2, mode='bilinear')], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(
            conv6, scale_factor=2, mode='bilinear')], dim=1))

        branch_conv_light = self.branch_conv_light(
            torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode='bilinear')], dim=1))
        branch_conv_heavy = self.branch_conv_heavy(
            torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode='bilinear')], dim=1))

        core_light = self.outc_light(F.interpolate(branch_conv_light, scale_factor=2, mode='bilinear'))
        core_heavy = self.outc_heavy(F.interpolate(branch_conv_heavy, scale_factor=2, mode='bilinear'))

        output_light= self.kernel_pred_light(data, core_light, white_level)
        output_heavy= self.kernel_pred_heavy(data, core_heavy, white_level)
        
        # print(uncertainty_map_light.shape)
        # uncertainty_map_light=uncertainty_map_light.unsqueeze(1)
        # uncertainty_map_heavy=uncertainty_map_heavy.unsqueeze(1)
        KPN2_conv1 = self.KPN2_conv1(
            torch.cat((data_with_est, core_light, core_heavy), 1))
        KPN2_conv2 = self.KPN2_conv2(F.avg_pool2d(KPN2_conv1, kernel_size=2, stride=2))
        KPN2_conv3 = self.KPN2_conv3(F.avg_pool2d(KPN2_conv2, kernel_size=2, stride=2))
        KPN2_conv4 = self.KPN2_conv4(F.avg_pool2d(KPN2_conv3, kernel_size=2, stride=2))
        KPN2_conv5 = self.KPN2_conv5(F.avg_pool2d(KPN2_conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        KPN2_conv6 = self.KPN2_conv6(torch.cat([KPN2_conv4, F.interpolate(
            KPN2_conv5, scale_factor=2, mode='bilinear')], dim=1))
        KPN2_conv7 = self.KPN2_conv7(torch.cat([KPN2_conv3, F.interpolate(
            KPN2_conv6, scale_factor=2, mode='bilinear')], dim=1))
        KPN2_conv8 = self.KPN2_conv8(torch.cat([KPN2_conv2, F.interpolate(
            KPN2_conv7, scale_factor=2, mode='bilinear')], dim=1))
        # return channel K*K*N
        KPN2_core = self.KPN2_outc(F.interpolate(
            KPN2_conv8, scale_factor=2, mode='bilinear'))
        

        output_light = output_light.unsqueeze(0)
        output_heavy = output_heavy.unsqueeze(0)
        output= self.KPN2_kernel_pred(torch.cat((output_light, output_heavy), 1), KPN2_core, white_level)
        # output = self.KPN2_kernel_pred(output_light, KPN2_core, white_level)
        output=output.unsqueeze(0)
        output = self.KPN2_conv9(output)

        return output


class DenoiseLoss(nn.Module):
    def __init__(self, n, hard_mining=0, norm=False):
        super(DenoiseLoss, self).__init__()
        self.n = n
        assert(hard_mining >= 0 and hard_mining <= 1)
        self.hard_mining = hard_mining
        self.norm = norm

    def forward(self, x, y):
        loss = torch.pow(torch.abs(x - y), self.n) / self.n
        if self.hard_mining > 0:
            loss = loss.view(-1)
            k = int(loss.size(0) * self.hard_mining)
            loss, idcs = torch.topk(loss, k)
            y = y.view(-1)[idcs]

        loss = loss.mean()
        if self.norm:
            norm = torch.pow(torch.abs(y), self.n)
            norm = norm.data.mean()
            loss = loss / norm
        return loss


class Loss(nn.Module):
    def __init__(self, n, hard_mining=0, norm=False):
        super(Loss, self).__init__()
        self.loss = DenoiseLoss(n, hard_mining, norm)

    def forward(self, x, y):
        z = []
        for i in range(len(x)):
            z.append(self.loss(x[i], y[i]))
        return z


if __name__ == '__main__':
    kpn = KPN(True, 1, True, [5], False, False,
              False, 'bilinear', False)
    # pretrained_net = torch.load(
    #     "/mnt/nvme/yihao/MyDataset-Color-KPN/models/KPN_single_image_epoch_20_bs_1_noise_no_test_Multi_Head_ImageNet_resnet50_PGD_combine_13_eps.pth")
    # load_dict(kpn, pretrained_net)
    print('Generator is loaded!')

    if True:
        for i in kpn.parameters():
            i.requires_grad = False

    attention_model = Atten_model(kpn)

    a = torch.randn(4, 3, 224, 224)
    b = attention_model(a, a)

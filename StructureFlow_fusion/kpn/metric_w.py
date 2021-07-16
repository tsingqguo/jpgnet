import torch
import kpn.utils as utils
import numpy as np
from PIL import Image
import cv2

height, width = 256, 256
N = 3
index = 1

mask = cv2.imread('./data/mask/00000.png')
# mask = cv2.resize(mask, (256, 256))
# mask = mask.transpose(2, 0, 1)
# mask = mask[0]/255

def write(img_list, name_list):
    utils.save_sample_png(sample_folder='./result/metric', sample_name='%d' % index, img_list=img_list,
                          name_list=name_list, pixel_max_cnt=255, height=height, width=width)


def visualization(input, core, pre):
    global index

    # arr = core.data.cpu().numpy()
    # np.save('./result/metric/{}_metric.data'.format(index), arr)

    core = core.view(1, N, -1, height, width)

    # L1 = core.norm(p=1, dim=2)
    # L2 = core.norm(p=2, dim=2)
    # max_, _ = core.max(dim=2)
    ave = core.sum(dim=2) / core.size(2)



    # write([L1], ['L1'])
    # write([L2], ['L2'])
    # write([max_], ['Max'])
    write([ave], ['Ave'])

    write([input], ['input'])
    write([pre], ['pre'])


    save_data(ave[0])


    index += 1


def save_channel(img_):
    img_ = np.clip(img_*255, 0, 255).astype(np.uint8)

    Image.fromarray(img_[0]).save('./result/metric/{}_c_0.jpg'.format(index))
    Image.fromarray(img_[1]).save('./result/metric/{}_c_1.jpg'.format(index))
    Image.fromarray(img_[2]).save('./result/metric/{}_c_2.jpg'.format(index))

    merge = img_[2] + img_[1] + img_[0]
    Image.fromarray(merge).save('./result/metric/{}_merge_c.jpg'.format(index))


    merge_ = np.copy(merge)
    merge_[merge_ < 100] = 0
    Image.fromarray(merge_).save('./result/metric/{}_merge_c_th100.jpg'.format(index))

    merge_[merge_ < 150] = 0
    Image.fromarray(merge_).save('./result/metric/{}_merge_c_th150.jpg'.format(index))

    Image.fromarray(img_.transpose(1, 2, 0)).save('./result/metric/{}_img.jpg'.format(index))

def save_ave(img_, mask, c):
    correct = img_[mask != 1]
    wrong = img_[mask == 1]
    msg = "channel:{} correct:{} uncertainty:{}".format(c, np.mean(correct), np.mean(wrong), 4)

    ff = open('./result/metric/{}_data.txt'.format(index), 'w')
    ff.write(msg)
    ff.close()


def save_data(ave_):
    img = ave_.data.cpu().numpy()

    save_channel(img)

    save_ave(img[0], mask, 0)
    save_ave(img[1], mask, 1)
    save_ave(img[2], mask, 2)
    save_ave(img[0] + img[1] + img[2], mask, 3)






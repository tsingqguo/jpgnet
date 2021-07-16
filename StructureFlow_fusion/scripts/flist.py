import os
import argparse
import numpy as np

# parser = argparse.ArgumentParser()
# # parser.add_argument('--path', default='../data/mask', type=str, help='path to the dataset')
# # parser.add_argument('--output', default='../data/list/irregular_mask.txt', type=str, help='path to the file list')
#
# # parser.add_argument('--path', default='../data/input', type=str, help='path to the dataset')
# # parser.add_argument('--output', default='../data/list/places_gt_train.txt', type=str, help='path to the file list')
#
# parser.add_argument('--path', default='../data/smooth', type=str, help='path to the dataset')
# parser.add_argument('--output', default='../data/list/places_structure_train.txt', type=str, help='path to the file list')
#
# args = parser.parse_args()

ext = {'.jpg', '.png'}
def generate(input_path, output_path):
    print(os.getcwd())
    images = []
    for root, dirs, files in os.walk(input_path):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1] in ext:
                images.append(os.path.join(root[1:], file))

    images = sorted(images)
    np.savetxt(output_path, images, fmt='%s')


# mask
generate('../data/mask/20', '../data/irregular_mask.txt')

# structure
generate('../data/train/smooth', '../data/train/places_structure_train.txt')

# imag
generate('../data/train/input', '../data/train/places_gt_train.txt')
#------------------------------------------------------------------------------------------------eval
# structure
generate('../data/eval/smooth', '../data/eval/places_structure_train.txt')

# imag
generate('../data/eval/input', '../data/eval/places_gt_train.txt')

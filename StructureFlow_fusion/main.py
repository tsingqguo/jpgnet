import os
import torch
import argparse
import shutil 
from src.config import Config
from src.structure_flow import StructureFlow

def main(mode=None):
    r"""starts the model
    Args:
        mode : train, test, eval, reads from config file if not specified
    """

    config = load_config(mode)
    config.MODE = mode
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(e) for e in config.GPU)

    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    model = StructureFlow(config)
    
    if mode == 'train':
        # config.print()
        print('\nstart training...\n')
        model.fusion_train()

    elif mode == 'test':
        print('\nstart test...\n')
        model.fusion_test()

    elif mode == 'eval':
        print('\nstart eval...\n')
        model.eval()        


def load_config(mode=None):
    r"""loads model config 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='places', type=str, help='output model name.')
    parser.add_argument('--config', type=str, default='model_config.yaml', help='Path to the config file.')
    parser.add_argument('--path', type=str, default='./results_struct', help='outputs path')
    parser.add_argument("--resume_all", default=True, action="store_true", help='load model from checkpoints')
    parser.add_argument("--remove_log", action="store_true", help='remove previous tensorboard log files')


    if mode == 'test':
        parser.add_argument('--input', default='./data/list/places_gt_train.txt', type=str, help='path to the input image files')
        parser.add_argument('--mask', default='./data/list/irregular_mask.txt', type=str, help='path to the mask files')
        parser.add_argument('--structure', default='./data/list/places_structure_train.txt', type=str, help='path to the structure files')
        parser.add_argument('--output', default='./test_result', type=str, help='path to the output directory')
        parser.add_argument('--model', type=int, default=3, help='which model to test')
    
    opts = parser.parse_args()
    config = Config(opts, mode)
    output_dir = os.path.join(opts.path, opts.name)
    perpare_sub_floder(output_dir)

    if mode == 'train':
        config_dir = os.path.join(output_dir, 'config.yaml')
        shutil.copyfile(opts.config, config_dir)
    return config


def perpare_sub_floder(output_path):
    img_dir = os.path.join(output_path, 'images_train')
    if not os.path.exists(img_dir):
        print("Creating directory: {}".format(img_dir))
        os.makedirs(img_dir)

    img_dir = os.path.join(output_path, 'images_eval')
    if not os.path.exists(img_dir):
        print("Creating directory: {}".format(img_dir))
        os.makedirs(img_dir)

    img_dir = os.path.join(output_path, 'images_test')
    if not os.path.exists(img_dir):
        print("Creating directory: {}".format(img_dir))
        os.makedirs(img_dir)


    checkpoints_dir = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        print("Creating directory: {}".format(checkpoints_dir))
        os.makedirs(checkpoints_dir) 
    

if __name__ == "__main__":
    main()

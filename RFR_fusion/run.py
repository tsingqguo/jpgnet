import argparse
import os
from model import RFRNetModel
from dataset import Dataset
from torch.utils.data import DataLoader
import config

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_root', type=str, default='./data/places.txt')
    parser.add_argument('--train_mask_root', type=str, default='./data/20_mask.txt')
    parser.add_argument('--train_batch_size', type=int, default=1)  # 6 default

    parser.add_argument('--eval_data_root', type=str, default='./data/places.txt')
    parser.add_argument('--eval_mask_root', type=str, default='./data/20_mask.txt')
    parser.add_argument('--eval_batch_size', type=int, default=1)  # 6 default

    parser.add_argument('--test_data_root', type=str, default='./data/places.txt')
    parser.add_argument('--test_mask_root', type=str, default='./data/20_mask.txt')
    parser.add_argument('--test_batch_size', type=int, default=1)  # 6 default

    parser.add_argument('--n_threads', type=int, default=0)  # 6 default

    parser.add_argument('--model_save_path', type=str, default='./checkpoints')
    parser.add_argument('--result_save_path', type=str, default='./results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=450000)
    parser.add_argument('--model_path', type=str, default="./checkpoints/RFR_duhuang_g_113000.pth")
    # parser.add_argument('--model_path', type=str, default="")


    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RFRNetModel()
    if args.test:
        model.initialize_model(args.model_path, False)
        # model.cuda()
        dataloader = DataLoader(Dataset(args.test_data_root, args.test_mask_root, args.mask_mode, args.target_size, mask_reverse = config.mask_reverse, training=False), batch_size = 1)

        model.fusion_test(dataloader)
    else:
        model.initialize_model(args.model_path, True)
        # model.cuda()
        train_data = Dataset(args.train_data_root, args.train_mask_root, args.mask_mode, args.target_size, mask_reverse = config.mask_reverse, training=True)
        train_loader = DataLoader(train_data, batch_size = args.train_batch_size, shuffle = True, num_workers = args.n_threads)

        eval_data = Dataset(args.eval_data_root, args.eval_mask_root, args.mask_mode, args.target_size, mask_reverse = config.mask_reverse, training=False)
        eval_loader = DataLoader(eval_data, batch_size = args.eval_batch_size, shuffle = False, num_workers = args.n_threads)

        print('train datasize:{}  eval dataset:{}'.format(len(train_data), len(eval_data)))

        model.fusion_train(train_loader, eval_loader, args.model_save_path, args.finetune, args.num_iters)

if __name__ == '__main__':
    run()
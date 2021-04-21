import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=7, help='segmentation classes')
        parser.add_argument('--data_path', type=str, default='C:\\Users\\mamduhkk\\Desktop\\研究生\\dataset\\Deepglobe_GL', help='path to dataset where images store')
        parser.add_argument('--model_path', type=str,default='./saved_models/', help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, default='./logs/', help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, default='PPN', help='task name for naming saved model files and log files')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--sub_batch_size', type=int, default=4, help='batch size for using local image patches')
        parser.add_argument('--size_g', type=int, default=512, help='size (in pixel) for downsampled global image')
        parser.add_argument('--size_lp', type=int, default=512, help='size (in pixel) for cropped local image')
        parser.add_argument('--size_p', type=int, default=32, help='size (in pixel) for cropped global out local image')
        parser.add_argument('--path_weight', type=str, default="C:\\Users\\mamduhkk\\Desktop\\研究生\\PPN_code\\PPN_ours\\saved_models\\best.pth", help='name for global model path')
        parser.add_argument('--sp_l', type=bool, default=False, help='local self-paced')
        parser.add_argument('--debug', type=bool, default=True, help='reinforcement self-paced')
        parser.add_argument('--update_ep', type=int, default=2, help='epoch of self-paced weights updating')
        parser.add_argument('--lamda_l', type=float, default=0.15)
        parser.add_argument('--lamda_rein', type=float, default=0.1)
        parser.add_argument('--mu', type=float, default=0.99)
        parser.add_argument("--local_rank", type=int, help="")

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr
        args.epochs_local = 0 # for pre-training local
        args.lr0 = 5e-5
        args.epochs_global = 0 # for pre-training global
        args.lr1 = 1e-4

        args.epochs_L = 1 # for classifier and reinforcement process
        args.lr2 = 1e-4
        args.lr3 = 1e-4

        return args

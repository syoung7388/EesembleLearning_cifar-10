from common.model import ResNet, Efficientnet
from common.utils import getdata
from common.utils import mini_batch_training
import argparse
import os
import torch 
import random 
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help= 'batch_size')
parser.add_argument('--resize', type=int, default=224, help= 'resize')
parser.add_argument('--lr', type=float, default=1e-5, help= 'lr')
parser.add_argument('--weight_decay', type=float, default=5e-4, help= 'weight_decay')
parser.add_argument('--epochs', type=int, default=20, help= 'epochs')
parser.add_argument('--gpus', type=str, default='0', help= 'epochs')
parser.add_argument('--model_name', type=str, default='resnet34', help= 'model_name')
parser.add_argument('--ckpt', type=str, default='', help= 'ckpt')
parser.add_argument('--root_dir', type=str, default='drive/app/cifar10/', help= 'data_dir')
parser.add_argument('--default_directory', type=str, default= 'drive/app/torch/save_models', help= 'default_directory')
parser.add_argument('--seed', type=int, default= '9712', help= 'seed')

args = parser.parse_args()
#[SEED]
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

#[GPU]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    #[DATA]
    train_dataloader, test_dataloader = getdata(args.root_dir, args.batch_size,  args.resize)


    #[MODEL]
    if 'resnet' in args.model_name:
        model = ResNet(args.model_name).to(device)
        params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
        classifier_param = model.model.fc.parameters()
    elif 'efficientnet' in args.model_name:
        model = Efficientnet(args.model_name).to(device)
        params_1x = [param for name, param in model.named_parameters() if 'classifier.1' not in str(name)]
        classifier_param = model.model.classifier[1].parameters() 
    start_time = time.time()
    #[TRAIN]
    mini_batch_training(model, train_dataloader, test_dataloader, args, device, classifier_param, params_1x)
    end_time = time.time()
    
    #[Train Info]
    info = open(f'./results/{args.model_name}/info.txt', 'w')
    info.write(f'train seconds: {end_time - start_time}\n')
    info.write(f'cuda version: {torch.version.cuda}\n')
    info.write(f'torch version: {torch.__version__}\n')
    info.close()
    
    
    




    


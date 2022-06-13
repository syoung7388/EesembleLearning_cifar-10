from unittest import TestResult
from common.model import ResNet, Efficientnet
import os 
import torch 
import argparse
from glob import glob
from common.utils import getdata
import torchvision
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
from itertools import combinations
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default='0', help= 'epochs')
parser.add_argument('--seed', type=int, default=9712, help= 'seed')
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

#[Data]
_, test_dataloader = getdata('drive/app/cifar10/', 1, 224)

#[Ensemble Comb]
file_names = [ 'resnet50', 'resnet101', 'resnet152', 'efficientnet_b1', 'efficientnet_b3', 'efficientnet_b5']

#[Ensemble method]
def hard_ensemble(preds):
    hard_answer = [0]*10
    hard_dis = []
    for i, pred in enumerate(preds):
        sort_pred, idx= torch.sort(pred, descending=True)
        hard_answer[idx[:, 0].item()] += 1
        hard_dis.append([sort_pred[:, 0].item() - sort_pred[:, 1].item(), idx[:, 0].item(), i]) #dis, pred_idx, model_num


    if max(hard_answer) == 1:
        hard_dis.sort(reverse = True)
        comb_log.write(f"dis: {hard_dis[0][0]:.3f}, predict: {hard_dis[0][1]}, model: {comb[hard_dis[0][2]]}\n")
        return hard_dis[0][-2]

    results =hard_answer.index(max(hard_answer))
    return results
        
    
    

    

tot_result = {"ens":[], "hard":[], "soft":[]}
for comb in file_comb:
    print(comb)
    models = []
    
    #[Log]
    log_dir = f"{comb[0]}_{comb[1]}_{comb[2]}"
    if not os.path.exists(f"./ensemble_result/{log_dir}"):
        os.makedirs(f"./ensemble_result/{log_dir}")
    result_pd = {"soft":[], "hard":[], "labels":[]}
    comb_log = open(f"./ensemble_result/{log_dir}/log.txt", "a")

    for f in comb:
        model_name = f
        if 'resnet' in f:
            print(f)
            model = ResNet(model_name).to(device)
        else:
            model = Efficientnet(model_name).to(device)

            
        model_root = glob(f'./experiment_results/{f}/model/best*')
        print(model_root)
        if len(model_root) == 0: continue
        model_root = model_root[0]
        model.load_state_dict(torch.load(model_root))
        model.eval()
        models.append(model)
        
        
    with torch.no_grad():    
        hard_sc = 0
        soft_sc = 0
        with tqdm(total=len(test_dataloader), desc = "Test") as pbar:
            for data, labels in test_dataloader:
                data = data.to(device)
                labels= labels.to(device)
                preds = []
                soft_answer = torch.zeros(1, 10).to(device)
                for model in models:
                    pred = model(data)
                    preds.append(pred)
                    soft_answer += pred
                hard_ans = hard_ensemble(preds)
                soft_ans = torch.argmax(soft_answer).item()
                
                hard_sc += 1 if hard_ans == labels.item() else 0
                soft_sc += 1 if soft_ans == labels.item() else 0
                result_pd["hard"].append(hard_ans)
                result_pd["soft"].append(soft_ans)
                result_pd["labels"].append(labels.item())
                pbar.update(1)
        
    hard_test_acc = (hard_sc / len(test_dataloader))*100 
    soft_test_acc = (soft_sc / len(test_dataloader))*100 
    
    tot_result["ens"].append(comb)
    tot_result["hard"].append(hard_test_acc)
    tot_result["soft"].append(soft_test_acc)

    print(f'hard voting:{hard_test_acc:.3f}, soft voting:{soft_test_acc:.3f}')
    comb_log.write(f'hard voting:{hard_test_acc:.3f}, soft voting:{soft_test_acc:.3f}')
    result_pd =  pd.DataFrame(result_pd)
    result_pd.to_csv(f'./ensemble_result/{log_dir}/result_pd.csv')
    comb_log.close()




max_s = max(tot_result["soft"])
max_s_idx = tot_result["soft"].index(max_s)
max_h = max(tot_result["hard"])
max_h_idx = tot_result["hard"].index(max_h)

log = open(f"./ensemble_result/log.txt", "a")
a = tot_result["ens"][max_s_idx]
b = tot_result["ens"][max_h_idx]
log.write(f"soft: {max_s}, {a}/n")
log.write(f"hard: {max_h}, {b}/n")
log.close()

tot_result = pd.DataFrame(tot_result)
tot_result.to_csv(f'./ensemble_result/tot_result.csv')





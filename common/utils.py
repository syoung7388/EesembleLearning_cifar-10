
from cgi import test
from turtle import color
import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os
import pandas as pd



def getdata(root_dir, batch_size, resize):

    mean = [0.4919, 0.4826, 0.4470]
    stev = [0.2408, 0.2373, 0.2559]


    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize, resize)), 
        torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean, stev)
    ])
    test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, stev)
        ])

    train_dataset = datasets.CIFAR10(root = root_dir, train = True, transform = train_transforms, download = True)
    test_dataset = datasets.CIFAR10(root = root_dir, train = False, transform = test_transforms, download = True)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_dataloader, test_dataloader
def draw_color_cell(x,color):
    color = f'background-color:{color}'
    return color
def get_result_save_dir(args):

    log_dir =f"/NasData/home/ksy/2022-1/Github/EnsembleLearning/experiment_results/{args.ckpt+args.model_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(log_dir+'/model')
    return log_dir

def mini_batch_training(model,train_dataloader, test_dataloader,  args, device, classifier_param, params_1x):


    #[Save Dir]
    log_dir = get_result_save_dir(args)

    #[Optimizer]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{'params': params_1x}, {'params':classifier_param, 'lr':args.lr*10}], lr=args.lr, weight_decay = args.weight_decay)
    start = time.time()


    results = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
    for e in range(1, args.epochs+1):
        log = open(f'{log_dir}/log.txt', "a")
        print(f'============={e}/{args.epochs}=============');log.write(f'============={e}/{args.epochs}=============\n')
        #[Train]
        model.train()
        train_loss = torch.tensor(0., device = device)
        train_accuracy = torch.tensor(0., device = device)
        with tqdm(total = len(train_dataloader), desc ="training") as pbar:
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_loss += loss*train_dataloader.batch_size
                    train_accuracy += (torch.argmax(preds, dim = 1) == labels).sum()
                pbar.update(1)

        #[Test]
        model.eval()
        test_loss = torch.tensor(0., device = device)
        test_accuracy = torch.tensor(0., device = device)

        with torch.no_grad():
            with tqdm(total=len(test_dataloader), desc = "Test") as pbar:
                for inputs, labels in test_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    preds = model(inputs)
                    loss = criterion(preds, labels)

                    test_loss += loss*test_dataloader.batch_size
                    test_accuracy += (torch.argmax(preds, dim = 1) == labels).sum()
                    pbar.update(1)


        train_loss = train_loss / len(train_dataloader.dataset)
        train_accuracy = (train_accuracy/len(train_dataloader.dataset))*100 
        print(f'[Training] loss: {train_loss:.2f}, accuracy: {train_accuracy:.3f}%'); log.write(f'[Training] loss: {train_loss:.2f}, accuracy: {train_accuracy:.3f}%\n')

        test_loss = test_loss / len(test_dataloader.dataset)
        test_accuracy = (test_accuracy/len(test_dataloader.dataset))*100 
        print(f'[Test] loss: {test_loss:.2f}, accuracy: {test_accuracy:.3f} %'); log.write(f'[Test] loss: {test_loss:.2f}, accuracy: {test_accuracy:.3f} %\n')


        #[Epoch Result Save]
        train_loss, train_accuracy, test_loss, test_accuracy = train_loss.item(), train_accuracy.item(), test_loss.item(), test_accuracy.item()
        train_loss, train_accuracy, test_loss, test_accuracy = round(train_loss ,3), round(train_accuracy,3), round(test_loss,3), round(test_accuracy,3)
        results['train_loss'].append(train_loss); results['train_acc'].append(train_accuracy); results['test_loss'].append(test_loss); results['test_acc'].append(test_accuracy)
        torch.save(model.state_dict(), log_dir+f'/model/{e}.pth')
        log.close()


    #[Tot Result Save]
    max_model_acc = max(results['test_acc'])
    max_model_epoch = results['test_acc'].index(max_model_acc) 
    os.rename(log_dir+f"/model/{max_model_epoch+1}.pth", log_dir+f"/model/best_{max_model_acc}_{max_model_epoch+1}.pth")
    result_pd = pd.DataFrame(results)
    result_pd.to_csv(log_dir+'/result.csv')


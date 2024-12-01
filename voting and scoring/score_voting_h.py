import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from timm.models import deit
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets
from tqdm import tqdm
import numpy as np

import argparse

from CNN_mod import CNN_mod, CNN_mod_deep
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from collections import OrderedDict, Counter

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Wafer Classification (03_INPUT_DATA)')
    parser.add_argument('--num_cls', type=int, default=3, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch Size')
    
    parser.add_argument('--model', default='cnn', help='mobilenetv2 | xception | deit_tiny | cnn | cnn_deep')
    parser.add_argument('--test_data_folder', default='/home/bsy/data/Textile/Textile_center/test', help = 'file_folder')
    parser.add_argument('--load_ckpt', default='/home/bsy/checkpoint/cnn_Textile_center_lr0.001.pth', help = 'checkpoint')


    return parser.parse_args()


class select_model(nn.Module):
    def __init__(self, num_cls, model_name):
        super(select_model, self).__init__()
        #self.model = timm.create_model('convnext_tiny', num_classes=num_cls)
        if model_name == 'mobilenetv2':
            self.model = timm.create_model('mobilenetv2_050', pretrained=True, num_classes=num_cls)
        elif model_name == 'xception':
            self.model = timm.create_model('xception', pretrained=True, num_classes=num_cls)
        elif model_name == 'deit_tiny':
            self.model = deit.deit_tiny_patch16_224(pretrained=True, num_classes=num_cls)
        elif model_name == 'cnn':
            self.model = CNN_mod()
        elif model_name == 'cnn_deep':
            self.model = CNN_mod_deep()
        else:
            raise ValueError("Invalid model choice. Please choose 'mobilenetv2' or 'xception' or 'deit_tiny'.")

    def forward(self, x):
        return self.model(x)


def transform_test():
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    
    return transform

# Load data
def dataset_load(opt, transform_test, batch_size):
    test_dataset = datasets.ImageFolder(opt.test_data_folder, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return test_loader, test_dataset

def weighted_f1_score(y_true, y_pred, weights=None):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    if weights is None:
        weights = [1] * len(f1_per_class)
    
    weights = weights.cpu().numpy()
    weighted_f1 = np.dot(f1_per_class, weights) / np.sum(weights)
    return weighted_f1

def find_common_elements(Labels, Preds, condition):
    filtered_vals = [(a.item(), b.item()) for a, b in zip(Labels, Preds) if condition(a, b)]
    item_counts = Counter(filtered_vals)
    if item_counts:
        h_freq = max(item_counts.values())
        return [item for item, count in item_counts.items() if count == h_freq]
    else:
        return []    


if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    print('==> Preparing data..')
    transform_test = transform_test()
    test_loader, test_dataset = dataset_load(args, transform_test=transform_test, batch_size=args.batch_size)

     
    train_numSample_list = test_dataset.targets
    num_cls = len(set(train_numSample_list))
    weights = [1-(train_numSample_list.count(i) / len(train_numSample_list)) for i in range(num_cls)]

    
    weights = torch.tensor(weights).to(device)

    checkpoint_path = args.load_ckpt  # 적절한 경로로 수정해주세요
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #print([k for k, v in checkpoint.items()]) ##### check
    
    model = select_model(num_cls=args.num_cls, model_name=args.model)
    #print(model.state_dict()) ##### check
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    #print(print([k for k, v in new_state_dict.items()])) ##### check)
    # 모델을 평가 모드로 설정
    model.eval()

    """checkpoint_path = args.load_ckpt  # 적절한 경로로 수정해주세요
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = select_model(num_cls=args.num_cls, model_name=args.model)
    
    model.load_state_dict(checkpoint)
    model = model.cuda()
    # 모델을 평가 모드로 설정
    model.eval()"""
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):  
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            vals = torch.bincount(outputs)
            pred_outputs = vals.eq(vals[0]).all()
            if pred_outputs:
                preds = [3]*args.batch_size
                
            else:
                _, a = torch.max(vals, 0)
                preds = [a.item()]*args.batch_size
                
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat([torch.tensor(p) for p in all_preds])
    all_preds = torch.cat(all_preds)
        
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    print("Confusion Matrix:")
    print(cm)
    correct_per_class = np.diag(cm)
    print(f"Correct predictions per class: {correct_per_class}")

    f1 = weighted_f1_score(all_labels, all_preds, weights)
    
    print(f'Weighted F1-score: {f1:.4f}')
        
    
    print('--- DONE! ---')

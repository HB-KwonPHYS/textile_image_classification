import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from timm.models import deit
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import numpy as np

import argparse
import wandb
from CNN_mod import CNN_mod, CNN_mod_deep

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Wafer Classification (03_INPUT_DATA)')
    parser.add_argument('--num_cls', type=int, default=3, help='number of classes')
    parser.add_argument('--epochs', type=int, default=30, help='EPOCHS')
    
    parser.add_argument('--lr', type=float, default=0.0245, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
    parser.add_argument('--optim', default='adam', help='adam | sgd')
    parser.add_argument('--schd', default='step', help='cosine | step')
    
    
    parser.add_argument('--model', default='mobilenetv2', help='mobilenetv2 | xception | deit_tiny | cnn | cnn_deep')
    parser.add_argument('--train_data_folder', default='/home/bsy/data/Textile/Textile_six/train', help = 'file_folder')
    parser.add_argument('--test_data_folder', default='/home/bsy/data/Textile/Textile_six/val', help = 'file_folder')
    parser.add_argument('--savename', type=str, default='Textile_6', help = 'Best Model Save Name')
    parser.add_argument('--model_save_dir', default='./checkpoint', help='Directory to save models')
    parser.add_argument('--WanDB_Subject', type=str, default='CNN', help='Setting WanDB Project Name')

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

def get_model_save_path(base_dir, model_name, save_name, lr):
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
    return os.path.join(base_dir, f"{model_name}_{save_name}_lr{lr}.pth")

def transform_train():
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2,
                                contrast=0.2, 
                                saturation=0.1, 
                                hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    
    return transform

def transform_test():
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    
    return transform


###add 231206
from torch.optim.lr_scheduler import LambdaLR
def warmup_lambda(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return float(epoch) / warmup_epochs
    return 1

# Load data
def dataset_load(opt, transform_train, transform_test, batch_size):

    
    train_dataset = datasets.ImageFolder(opt.train_data_folder, transform=transform_train)
    val_dataset = datasets.ImageFolder(opt.test_data_folder, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, train_dataset, val_dataset


# Training
def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss /= len(train_loader.dataset)
    train_acc = correct / total  # [0, 1] 사이
    print(f"Train Error: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    wandb.log({"train_loss": train_loss, "train_acc": train_acc}, step=epoch)
    


def validate(model, val_loader, criterion, epoch, device, best_acc, not_improved_count, patience):
    model.eval()
    val_loss, correct, total = 0, 0, 0
   
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    
    if val_acc > best_acc:
        print('Saving Best Model')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = get_model_save_path(args.model_save_dir, args.model, args.savename, args.lr)
        torch.save(model.state_dict(), save_path)
        best_acc = val_acc
        not_improved_count = 0  # reset the count
    else:
        not_improved_count += 1
    
    print(f"Test Error: \n Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    wandb.log({"test_loss": val_loss, "test_acc": val_acc}, step=epoch)
    
    if not_improved_count >= patience:
        print(f'Early stopping triggered after {patience} epochs without improvement.')
        return best_acc, not_improved_count, True  # return an additional flag to indicate early stopping
    
    return best_acc, not_improved_count, False  # no early stopping triggered



def run_training_and_validation(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, best_acc):
    not_improved_count = 0  # initialize
    patience = 5  # define your patience value
    warmup_epochs = 5  # Warm-up epoch
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(model, train_loader, criterion, optimizer, epoch, device)

        #### add 231206
        # Warm-up 
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        best_acc, not_improved_count, early_stop_flag = validate(model, val_loader, criterion, epoch, device, best_acc, not_improved_count, patience)
        scheduler.step()
        
        if early_stop_flag:
            break
        
    return best_acc



if __name__ == '__main__':
    args = parse_arguments()

    #
    config_ = {
    "LR": args.lr, 
    "Epochs": args.epochs,
    "Batch_size": args.batch_size,
    "Model" : args.model,
    "Save_pth" : args.savename
    }

    wandb.init(project=args.WanDB_Subject, config=config_) #""In_Domain, Domain_Adaptation, IMAGENET_FT

    
    # Data
    print('==> Preparing data..')
    transform_train = transform_train()
    transform_test = transform_test()
    train_loader, val_loader, train_dataset, val_dataset = dataset_load(args, transform_train=transform_train, transform_test=transform_test, batch_size=args.batch_size)
    
    model = select_model(num_cls=args.num_cls, model_name=args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    #
    train_numSample_list = train_dataset.targets
    num_cls = len(set(train_numSample_list))
    weights = [1-(train_numSample_list.count(i) / len(train_numSample_list)) for i in range(num_cls)]

    
    weights = torch.tensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Invalid optimizer choice. Please choose 'adam' or 'sgd'.")
    
    if args.schd == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.schd == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        raise ValueError("Invalid scheduler choice. Please choose 'cosine' or 'step'.")

    ### 231206 add
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_lambda(epoch, warmup_epochs=5))  
    
    best_acc = 0
    best_acc = run_training_and_validation(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, device, best_acc)
    print('--- DONE! ---')

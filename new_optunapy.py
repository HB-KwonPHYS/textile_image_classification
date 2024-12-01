import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import timm
import argparse
from timm.models import deit
from torch.utils.data import DataLoader

class select_model(nn.Module):
    def __init__(self, num_cls, model_name):
        super(select_model, self).__init__()
        if model_name == 'mobilenetv2':
            self.model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=num_cls)
        elif model_name == 'xception':
            self.model = timm.create_model('xception', pretrained=True, num_classes=num_cls)
        elif model_name == 'deit_tiny':
            self.model = deit.deit_tiny_patch16_224(pretrained=True, num_classes=num_cls)

    def forward(self, x):
        return self.model(x)
    
    
def parse_option():
    parser = argparse.ArgumentParser('OPTUNA for training')
    parser.add_argument('--model', default='cnn_deep', help='mobilenetv2 | xception | deit_tiny | cnn | cnn_deep')
    parser.add_argument('--train_data_folder', default='/home/bsy/data/Textile/Textile_six/train', help = 'file_folder')
    parser.add_argument('--test_data_folder', default='/home/bsy/data/Textile/Textile_six/val', help = 'file_folder')
    parser.add_argument('--num_cls', type=int, default=3, help='number of classes')
    opt, _ = parser.parse_known_args()
    
    return opt

opt = parse_option()


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
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
    train_acc = 100.0 * correct / total
    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


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


# Load data
def dataset_load(opt, transform_train, transform_test, batch_size):

    
    train_dataset = ImageFolder(opt.train_data_folder, transform=transform_train)
    val_dataset = ImageFolder(opt.test_data_folder, transform=transform_test)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, train_dataset, val_dataset


def objective(trial):
    # Define hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    scheduler_name = trial.suggest_categorical('scheduler', ['CosineAnnealingLR', 'StepLR'])
    
    # Load data
    trans_train = transform_train()
    trans_test = transform_test()
    
    train_loader, val_loader, train_dataset = dataset_load(opt, trans_train, trans_test, batch_size)
    
    
    # Create model
    model = select_model(num_cls=opt.num_cls, model_name=opt.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = nn.DataParallel(model)
    model = model.to(device)

    train_numSample_list = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    
    # Define loss function and optimizer
    weights = [1-(train_numSample_list.count(i) / len(train_numSample_list)) for i in range(opt.num_cls)]
    
    weights = torch.tensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    elif scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        

    # Training loop
    for epoch in range(30):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    return val_loss


if __name__ == '__main__':
    pruner_ = optuna.pruners.MedianPruner() #optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='minimize', pruner=pruner_)
    study.optimize(objective, n_trials=10)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Best value: {study.best_trial.value}")

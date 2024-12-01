import torch
import torch.nn as nn

class CNN_mod(nn.Module):
    # CNN_mode의 새로운 neural network class 정의(nn.Module 상속 받음)
    def __init__(self):
        super(CNN_mod, self).__init__()        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # 첫 번째 convolution layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 두 번째 convolution layer
        self.fc1 = nn.Linear(64 * 56 * 56, 64)
        # 첫 번째 fully connected layer: input neuron 수: 64x56x56, output neuron 수: 64
        self.fc2 = nn.Linear(64, 3)
        # 첫 번째 fully connected layer: input neuron 수: 64, output neuron 수: 3 (클래스 수)

    def forward(self, x):
        # forward method (x: input data)
        x = self.pool(torch.relu(self.conv1(x)))
        # x를 conv1 layer 통과 후 activation function relu 적용 --> pooling layer 통과
        x = self.pool(torch.relu(self.conv2(x)))
        # x를 conv2 layer 통과 후 activation function relu 적용 --> pooling layer 통과
        x = x.view(-1, 64 * 56 * 56)
        # x의 형태를 [batch size, -1]로 변환하여 fully connected layer에 input 으로 사용 가능하도록 만듦
        x = torch.relu(self.fc1(x))
        # x를 fc1 layer 통과 시켜서 activation function relu 적용
        x = self.fc2(x)
        # x를 fc2 layer 통과 시킴
        return x
        # output 반환



class CNN_mod_deep(nn.Module):
    def __init__(self):
        super(CNN_mod_deep, self).__init__()
        
        # Convolutional layers
        # 총 4개의 convolution layer 정의 (input channel 수, output channel 수, 커널 사이즈, 스트라이드, 패딩)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        # 2x2 사이즈의 max pooling을 정의하였으며, 여기서는 input의 space size를 절반으로 줄여줌
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected(fc) layers
        # 3개의 fc layer를 정의, (input unit 수, output unit 수)로 나타내며, 마지막 fc3 layer에서 output unit 수는 클래스의 개수 
        self.fc1 = nn.Linear(256 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 3)
        
        # Dropout to prevent overfitting
        # 학습하는 동안 랜덤으로 50%fmf 0으로 만들어서 overfitting 방지하기 위함
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 총 4개의 convolution layer + relu activation + max pooling layer 통과시키기
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        
        # tensor 형태로 변환하기 위함 (배치사이즈, -1) 로 변경 후 fc layer에 input 값으로 활용
        x = x.view(-1, 256 * 14 * 14)
        
        x = torch.relu(self.fc1(x)) #x를 fc1 layer 통과 시켜서 activation function relu 적용
        x = self.dropout(x) # 위에서 relu 적용 한 다음에 dropout 진행하여 x로 받음
        x = torch.relu(self.fc2(x)) # 위에서 수행한 값을 다시 fc2 layer 통과 시키고 activation function relu 적용
        x = self.dropout(x)  # 위에서 relu 적용 한 다음에 dropout 진행하여 x로 받음
        x = self.fc3(x) # x를 fc3 layer 통과시킴
        
        return x # output 반환

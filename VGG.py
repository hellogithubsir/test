# 导入库
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import time
from torchvision.datasets import CIFAR10
import torchvision.models as models


# 搭建VGG网络模型
## VGG卷积重复模块搭建
def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
           nn.ReLU(True)]
    # 定义重复的中间层
    for i in range(num_convs - 1):
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
    # 定义池化层
    net.append(nn.MaxPool2d(2 , 2))
    return nn.Sequential(*net)#*代表返回对象克隆体

## VGG堆叠模块搭建
def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)

## 网络搭建
class VGG(nn.Module):
    def __init__(self, vgg_net):
        super(VGG, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


# 数据预处理函数：
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化
    x = x.transpose((2, 0, 1))  # 更改数据通道顺序，与模型输入通道顺序相同,即（通道数，长，宽）
    x = torch.from_numpy(x)
    return x


# 定义训练与测试函数：
def train_model(model, num_epochs, criterion, optimizer):
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        start = time.time()
        # 模型训练
        model.train()
        for data in train_data:
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            train_acc += torch.sum(preds == labels).item()
        # 模型测试
        model.eval()
        for data in test_data:
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            valid_loss += loss.data.item()
            valid_acc += torch.sum(preds == labels).item()
        end = time.time()
        print('Epoch {}/{}, Train Loss: {:.4f}, Train Acc: {}, Valid Loss: {:.4f}, Valid Acc: {}, Time {}'.format(
            epoch, num_epochs - 1, train_loss / len(train_data), train_acc / len(train_data),
                   valid_loss / len(test_data), valid_acc / len(test_data), end - start))


if __name__ == "__main__":
    # ## 打印模型
    vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
    vgg = VGG(vgg_net)
    print(vgg)
    ## 验证模型输入输出尺寸
    # input_demo = Variable(torch.ones(1, 3, 227, 227))
    # output_demo = VGG(input_demo)
    # print(output_demo.shape)

    # 加载训练和测试集数据：
    train_set = CIFAR10('./data', train=True, transform=data_tf)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('./data', train=False, transform=data_tf)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    # 损失函数与优化器：
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg.parameters(), lr=1e-1)

    # 训练20次：
    train_model(vgg, 20, criterion, optimizer)
    # todo 保存/调用模型
    save_path = "./AlexNet.pkl"
    ## 模型与参数一起保存
    torch.save(vgg, save_path)  # model.state_dict()仅保存参数，重新使用时
    seq_net1 = torch.load(save_path)
    ## 仅保存模型参数，需要重新建立模型才可使用
    torch.save(vgg.state_dict(), save_path)

    seq_net1 = vgg()
    seq_net1.load_state_dict(torch.load(save_path))
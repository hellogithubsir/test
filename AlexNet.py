# 导入库
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import time
from torchvision.datasets import CIFAR10
from torchvision import models
from tqdm import tqdm

# 搭建AlexNet网络模型
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层是卷积层：输入通道是3，输出通道是64,卷积核大小是5x5，步长为1,没有填充
        # 激活函数ReLU参数设置：
        # True表示直接对输入进行修改，False表示创建新创建一个对象进行修改，默认False。
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU()
        )
        # 第二层为最大池化层：大小为3x3，步长为2，没有填充
        self.max_pool1 = nn.MaxPool2d(3, 2)
        # 第三层是卷积层：输入的通道是64，输出的通道是64，没有填充
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1),
            nn.ReLU(True)
        )
        # 第四层是最大池化层：大小为3x3，步长是 2，没有填充
        self.max_pool2 = nn.MaxPool2d(3, 2)
        # 第五层是全连接层：输入是 1204 ，输出是384
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 384),
            nn.ReLU(True)
        )
        # 第六层是全连接层：输入是 384， 输出是192
        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(True)
        )
        # 第七层是全连接层，输入是192， 输出是10
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        # 将图片矩阵拉平
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 数据预处理函数：
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化
    x = x.transpose((2, 0, 1)) #更改数据通道顺序，与模型输入通道顺序相同,即（通道数，长，宽）
    x = torch.from_numpy(x)
    return x
# 定义训练与测试函数：
# def train_model(model, num_epochs ,criterion, optimizer):
#     for epoch in range(num_epochs):
#         train_loss = 0.0
#         train_acc = 0.0
#         valid_loss = 0.0
#         valid_acc = 0.0
#         start = time.time()
#         #模型训练
#         model.train()
#         for data in train_data:
#             inputs, labels = data
#             inputs, labels = Variable(inputs),Variable(labels)
#             optimizer.zero_grad()
#             outputs = model(inputs.to(device))
#             _, preds = torch.max(outputs.data, 1)
#             loss = criterion(outputs, labels.to(device))
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.data.item()
#             train_acc += torch.sum(preds == labels.to(device)).item()
#         #模型测试
#         model.eval()
#         for data in test_data:
#             inputs, labels = data
#             inputs, labels = Variable(inputs),Variable(labels)
#             optimizer.zero_grad()
#             outputs = model(inputs.to(device))
#             _, preds = torch.max(outputs.data, 1)
#             loss = criterion(outputs, labels.to(device))
#             valid_loss += loss.data.item()
#             valid_acc += torch.sum(preds == labels.to(device)).item()
#         end = time.time()
#         print('Epoch {}/{}, Train Loss: {:.4f}, Train Acc: {}, Valid Loss: {:.4f}, Valid Acc: {}, Time {}'.format(
#         epoch, num_epochs - 1,train_loss/len(train_data),train_acc/len(train_data),
#         valid_loss/len(test_data),valid_acc/len(test_data),end - start))

def train_model(model, num_epochs ,criterion, optimizer):
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        start = time.time()
        #模型训练
        model.train()
        train_bar = tqdm(train_data)
        for step, data in enumerate(train_bar):
            inputs, labels = data
            inputs, labels = Variable(inputs),Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            train_acc += torch.sum(preds == labels.to(device)).item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     num_epochs,
                                                                     loss)
        # #模型测试
        model.eval()
        valid_bar = tqdm(test_data)
        with torch.no_grad():
            for step, data in enumerate(valid_bar):
                inputs, labels = data
                inputs, labels = Variable(inputs),Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels.to(device))
                valid_loss += loss.data.item()
                valid_acc += torch.sum(preds == labels.to(device)).item()
                valid_bar.desc = "test epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         num_epochs,
                                                                         loss)
        end = time.time()



        print('Epoch {}/{}, Train Loss: {:.4f}, Train Acc: {}, Valid Loss: {:.4f}, Valid Acc: {}, Time {}'.format(
            epoch+1,
            num_epochs ,
            train_loss/len(train_data),
            train_acc/len(train_data),
            valid_loss/len(test_data),
            valid_acc/len(test_data),end - start))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## 加载与打印模型
    alexnet = AlexNet()#注意变量名不要模型名一致，否则保存报错
    alexnet.to(device)
    print(alexnet)
    ## 验证模型输入输出尺寸
    # input_demo = Variable(torch.ones(1, 3, 227, 227))
    # output_demo = AlexNet(input_demo)
    # print(output_demo.shape)

    # 加载训练和测试集数据：
    train_set = CIFAR10('./data', train=True, transform=data_tf)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('./data', train=False,  transform=data_tf)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    # 损失函数与优化器：
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(alexnet.parameters(), lr= 1e-1)

    # 训练20次：
    train_model(alexnet, 20, criterion, optimizer)

    # todo 保存/调用模型
    save_path = "./AlexNet.pkl"
    ## 模型与参数一起保存
    torch.save(AlexNet, save_path)  # model.state_dict()仅保存参数，重新使用时
    seq_net1 = torch.load(save_path)
    ## 仅保存模型参数，需要重新建立模型才可使用
    torch.save(AlexNet.state_dict(), save_path)

    seq_net1 = AlexNet()
    seq_net1.load_state_dict(torch.load(save_path))


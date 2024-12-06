# 导入库
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import time
from torchvision.datasets import CIFAR10
from torchvision import transforms
'''
https://blog.csdn.net/wsjjason/article/details/113620606【每一步都标记了kernel详细信息】
https://zhuanlan.zhihu.com/p/52802896【说明了辅助分类器详细的加入方式】
https://blog.csdn.net/weixin_38132153/article/details/107660114[Block类里的block3的kernelsize应该是5，padding是2]

'''


# 搭建googlenet网络模型
# Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, out_chanel_1, out_channel_3_reduce, out_channel_3,
                 out_channel_5_reduce, out_channel_5, out_channel_pool):
        super(Inception, self).__init__()
        # todo 看看这个block_list应该是可以去掉
        self.block1 = nn.Conv2d(in_channels=in_channels, out_channels=out_chanel_1, kernel_size=1)
        self.block2_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_3_reduce, kernel_size=1)
        self.block2 = nn.Conv2d(in_channels=out_channel_3_reduce, out_channels=out_channel_3, kernel_size=3, padding=1)
        self.block3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_5_reduce, kernel_size=1)
        self.block3 = nn.Conv2d(in_channels=out_channel_5_reduce, out_channels=out_channel_5, kernel_size=5, padding=2)
        self.block4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.block4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_pool, kernel_size=1)

        # self.incep = nn.Sequential(*block)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.block2_1(x))
        out3 = self.block3(self.block3_1(x))
        out4 = self.block4(self.block4_1(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)

        print(out1.shape,out2.shape,out3.shape,out4.shape,out.shape)
        return out


# 辅助分类器(在完整网络中间某层输出结果以一定的比例添加到最终结果分类)
class AuxiliaryClassifiction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AuxiliaryClassifiction, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.linear1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.7)
        self.linear2 = nn.Linear(in_features=1024, out_features=out_channels)

    def forward(self, x):
        x = self.conv1(self.avgpool(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        out = self.linear2(self.dropout(x))
        return out


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, stage='train'):
        super(GoogleNet, self).__init__()
        self.stage = stage

        self.blockA = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(64),

        )
        self.blockB = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # 三个inception
        self.blockC = nn.Sequential(
            Inception(in_channels=192, out_chanel_1=64, out_channel_3_reduce=96, out_channel_3=128,
                      out_channel_5_reduce=16, out_channel_5=32, out_channel_pool=32),
            Inception(in_channels=256, out_chanel_1=128, out_channel_3_reduce=128, out_channel_3=192,
                      out_channel_5_reduce=32, out_channel_5=96, out_channel_pool=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.blockD_1 = Inception(in_channels=480, out_chanel_1=192, out_channel_3_reduce=96, out_channel_3=208,
                                  out_channel_5_reduce=16, out_channel_5=48, out_channel_pool=64)
        # 第1个辅助分类器
        if self.stage == 'train':
            self.Classifiction_logits1 = AuxiliaryClassifiction(in_channels=512, out_channels=num_classes)

        self.blockD_2 = nn.Sequential(
            Inception(in_channels=512, out_chanel_1=160, out_channel_3_reduce=112, out_channel_3=224,
                      out_channel_5_reduce=24, out_channel_5=64, out_channel_pool=64),
            Inception(in_channels=512, out_chanel_1=128, out_channel_3_reduce=128, out_channel_3=256,
                      out_channel_5_reduce=24, out_channel_5=64, out_channel_pool=64),
            Inception(in_channels=512, out_chanel_1=112, out_channel_3_reduce=144, out_channel_3=288,
                      out_channel_5_reduce=32, out_channel_5=64, out_channel_pool=64),
        )
        # 第2个辅助分类器
        if self.stage == 'train':
            self.Classifiction_logits2 = AuxiliaryClassifiction(in_channels=528, out_channels=num_classes)

        self.blockD_3 = nn.Sequential(
            Inception(in_channels=528, out_chanel_1=256, out_channel_3_reduce=160, out_channel_3=320,
                      out_channel_5_reduce=32, out_channel_5=128, out_channel_pool=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.blockE = nn.Sequential(
            Inception(in_channels=832, out_chanel_1=256, out_channel_3_reduce=160, out_channel_3=320,
                      out_channel_5_reduce=32, out_channel_5=128, out_channel_pool=128),
            Inception(in_channels=832, out_chanel_1=384, out_channel_3_reduce=192, out_channel_3=384,
                      out_channel_5_reduce=48, out_channel_5=128, out_channel_pool=128),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.blockA(x)
        x = self.blockB(x)
        x = self.blockC(x)
        Classifiction1 = x = self.blockD_1(x)
        Classifiction2 = x = self.blockD_2(x)
        x = self.blockD_3(x)
        out = self.blockE(x)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.stage == 'train':
            Classifiction1 = self.Classifiction_logits1(Classifiction1)
            Classifiction2 = self.Classifiction_logits2(Classifiction2)
            return Classifiction1, Classifiction2, out
        else:
            return out


# 数据预处理函数：
def data_tf(x):
    x = x.resize((224, 224), 2)#input需要(8, 3, 224, 224),CIFAR是([64, 3, 32, 32])，要resize
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

# 定义训练与测试函数：
def train_model(model, num_epochs ,criterion, optimizer):
    for epoch in range(num_epochs):
        print('the num_epochs is {}'.format(epoch))
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        start = time.time()
        #模型训练
        model.stage = 'train' #因为是循环所以这里要设置回来，否则eval中是没有两个分类器，输出是对不上的
        model.train()
        for data in train_data:
            inputs, labels = data
            inputs, labels = Variable(inputs),Variable(labels)
            # print(inputs.shape)
            optimizer.zero_grad()
            # outputs = model(inputs.to(device))
            Classifiction1, Classifiction2, outputs = model(inputs.to(device))
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.to(device)) \
                   + 0.3 * criterion(Classifiction1, labels.to(device)) \
                   + 0.3 * criterion(Classifiction2, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            train_acc += torch.sum(preds == labels.to(device)).item()
        #更改属性,在推理是时是没有两个辅助分类器的
        model.stage = 'eval'
        #模型测试
        model.eval()
        for data in test_data:
            inputs, labels = data
            inputs, labels = Variable(inputs),Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.to(device))
            valid_loss += loss.data.item()
            valid_acc += torch.sum(preds == labels.to(device)).item()
        end = time.time()
        print('Epoch {}/{}, Train Loss: {:.4f}, Train Acc: {}, Valid Loss: {:.4f}, Valid Acc: {}, Time {}'.format(
        epoch, num_epochs - 1,train_loss/len(train_data),train_acc/len(train_data),
        valid_loss/len(test_data),valid_acc/len(test_data),end - start))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## 打印模型- debug
    ggnet = GoogleNet(num_classes=10, stage='train')  # 注意变量名不要模型名一致，否则保存报错
    ggnet.to(device)
    # input = torch.randn(8, 3, 224, 224)
    # Classifiction1, Classifiction2, out = ggnet(input.to(device))
    # print(Classifiction1.shape, Classifiction2.shape, out.shape)
    # print(Classifiction1, 0.3 * Classifiction1)

    # # 加载训练和测试集数据：
    train_set = CIFAR10('./data', train=True, transform=data_tf)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('./data', train=False, transform=data_tf)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    # # 损失函数与优化器：
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ggnet.parameters(), lr=1e-1)

    # 训练20次：
    train_model(ggnet, 20, criterion, optimizer)
#
#     # todo 保存/调用模型
#     save_path = "./AlexNet.pkl"
#     ## 模型与参数一起保存
#     torch.save(GoogleNet, save_path)  # model.state_dict()仅保存参数，重新使用时
#     seq_net1 = torch.load(save_path)
#     ## 仅保存模型参数，需要重新建立模型才可使用
#     torch.save(GoogleNet.state_dict(), save_path)
#
#     seq_net1 = GoogleNet()
#     seq_net1.load_state_dict(torch.load(save_path))
#

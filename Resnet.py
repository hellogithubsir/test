'''

https://blog.csdn.net/frighting_ing/article/details/121324000[图好]
https://zhuanlan.zhihu.com/p/474790387【官方代码详细解释】
https://blog.csdn.net/weixin_61785507/article/details/125224327【使用预训练在自己项目时类别数目报错】
https://blog.csdn.net/weixin_43818631/article/details/121029153【预训练权重问题】
https://www.jianshu.com/p/7a7d45b8e0ee【pytorch加载模型和初始化权重】
https://blog.csdn.net/u013972657/article/details/115869102【使用resnet提取特征】
https://blog.csdn.net/cyj972628089/article/details/122799222[pytorch动态调节学习率]
'''

import torch
from torchvision.datasets import CIFAR10
from torch.autograd import Variable
import time
from torchvision import models
import torch.nn as nn
from torchvision import  transforms
from tqdm import tqdm
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
# 定义训练与测试函数：
def train_model(model, num_epochs ,criterion, optimizer):
    lr_list = []
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
            scheduler.step(loss)
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
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
        print("epoch={}, lr={}".format(epoch+1, optimizer.state_dict()['param_groups'][0]['lr']))
    print('training completed！',len(lr_list))#思考下782是啥
    plt.plot(range(782), lr_list, color='r', label='ReduceLROnPlateau') #todo 这里要看下怎么改
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    # 判断cuda? cpu?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    ## 加载与打印模型
    # net = models.resnext101_32x8d(num_classes = 10)
    net = models.resnet34(num_classes = 10)

    net.to(device)#网络要放在加载权重前，不然报错
    #加载预训练权重
    # pretrained_dict = torch.load('./data/resnext101_32x8d-8ba56ff5.pth', map_location=device)
    pretrained_dict = torch.load('./data/resnet34-pre.pth', map_location=device)

    model_dict = net.state_dict()
    # 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    # 更新权重
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # print(net)、
    # 数据处理
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 加载训练和测试集数据：
    train_set = CIFAR10('./data', train=True, transform=data_transform["train"])
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('./data', train=False, transform=data_transform["val"])
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    train_num = len(train_set)
    val_num = len(test_set)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 损失函数与优化器：
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # 训练20次：
    train_model(net, 100, criterion, optimizer)
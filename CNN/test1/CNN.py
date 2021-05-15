import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt





class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            # 卷积参数设置
            nn.Conv2d(
                in_channels=1,  # 输入数据的通道为1，即卷积核通道数为1
                out_channels=16,  # 输出通道数为16，即卷积核个数为16
                kernel_size=(2,300),  # 卷积核的尺寸为 5*5
                stride=1,  # 卷积核的滑动步长为1
                padding=2,  # 边缘零填充为2
            ),
            nn.ReLU(),  # 激活函数为Relu
            nn.MaxPool2d(2),  # 最大池化 2*2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3,300), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 36),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(36, 10)  # 最后输出结果个数为2
        self.soft = nn.Softmax()


    def forward(self,x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return self.soft(out)

        # self.filter_sizes = (2, 3, 4)
        # self.embed = 300
        # self.num_filters = 256
        # self.dropout = 0.5
        # self.conv1=nn.Conv2d(1,6,(k,300) for k in self.filter_sizes)
        # self.conv2=nn.Conv2d(3,12,[2,3,4])
        # self.conv3=nn.MaxPool2d(2)
        # self.conv4=nn.Softmax()

        # self.conv2=nn.MaxPool2d()
        # self.conv1= nn.Sequential(
        #     nn.Conv1d(300,100,4)
        #     nn.MaxPool1d(2)
        #     nn.ReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(100,36,4)
        #     nn.MaxPool1d(2)
        #     nn.ReLU()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(36,9,2)
        #     nn.Linear()
        # )




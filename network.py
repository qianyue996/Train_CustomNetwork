import torch
import torch.nn as nn

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )

    def forward(self, x):
        return self.convs(x)

class GlobalMaxPool2d(nn.Module):
    def __init__(self, k):
        super(GlobalMaxPool2d,self).__init__()
        self.gvp = nn.MaxPool2d(k)
    def forward(self, x):
        return self.gvp(x)

class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_layer1 = Conv_BN_LeakyReLU(3,32)
        self.conv_layer2 = Conv_BN_LeakyReLU(32,64)
        self.conv_layer3 = nn.Sequential(Conv_BN_LeakyReLU(64,128),Conv_BN_LeakyReLU(128,64),Conv_BN_LeakyReLU(64,128))
        self.conv_layer4 = nn.Sequential(Conv_BN_LeakyReLU(128,256),Conv_BN_LeakyReLU(256,128),Conv_BN_LeakyReLU(128,256))
        self.conv_layer5 = nn.Sequential(Conv_BN_LeakyReLU(256,512),Conv_BN_LeakyReLU(512,256),Conv_BN_LeakyReLU(256,512),Conv_BN_LeakyReLU(512,256),Conv_BN_LeakyReLU(256,512))
        self.conv_layer6 = nn.Sequential(Conv_BN_LeakyReLU(512,1024),Conv_BN_LeakyReLU(1024,512),Conv_BN_LeakyReLU(512,1024),Conv_BN_LeakyReLU(1024,512),Conv_BN_LeakyReLU(512,1024))
        
        self.head = nn.Sequential(nn.Conv2d(1024,100,1),GlobalMaxPool2d(7),nn.Softmax(1))

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.maxpool(x)
        x = self.conv_layer2(x)
        x = self.maxpool(x)
        x = self.conv_layer3(x)
        x = self.maxpool(x)
        x = self.conv_layer4(x)
        x = self.maxpool(x)
        feature = self.conv_layer5(x)
        x = self.maxpool(feature)
        x = self.conv_layer6(x)

        head_output = self.head(x).squeeze(-1).squeeze(-1)
        return head_output
    
if __name__=='__main__':
    input=torch.randn((1,3,224,224))
    # 查看Darknet网络
    net = Darknet19()
    output = net(input)
    print(output.shape)
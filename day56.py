#Build ResNet(Residual Network) from scratch
import torch
import torch.nn as nn
import torch.nn.functional as f
class BasicBlock(nn.Module):
    expansion=1
    def __init(self,in_planes,planes,stride=1):
        super(BasicBlock,self).__init__()
        #first convolution layer
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        #second convulation layer
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,adding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        #shortcut connection 
        self.shortcut=nn.Sequential()
        if stride!=1 or in_planes!=self.expansion *planes:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_planes,self.expansion * planes,kernel_size=1,stride=stride,padding=1,bias=False) ,
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self,x):
        identity=x
        out=f.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2)
        out+=self.shortcut(identity)
        out=f.relu(out)
        return out
#create dummy input
dummy_input = torch.randn(2, 3, 60 ,1,64)
block = BasicBlock(in_planes=3, planes=64, stride=2)
output = block(dummy_input)
print("Input Shape: ", dummy_input.shape)
print("Output Shape:", output.shape)
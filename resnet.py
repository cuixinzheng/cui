import torch
from torch import nn
from torch.nn import functional as F
class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self,ch_in,ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out !=ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self,x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #shor cut
        #extra module:[b,ch_in,h,w] =>[b,ch_out,h,w]
        #element_wise add:
        out = self.extra(x) + out
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16)
        )
        # follow 4 blocks
        #[b,64,h,w] => [b,128,h,w]
        self.blk1 = ResBlk(16,16)
        self.blk2 = ResBlk(16,32)
        self.blk3 = ResBlk(32,100)
        self.blk4 = ResBlk(100,10)
        self.outlayer = nn.Linear(10*32*32,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)
        return x

def main():

    blk = ResBlk(64,128)
    tmp = torch.randn(2,64,32,32)
    out = blk(tmp)
    print('blk:',out.shape)
    model = ResNet18()
    out = model(tmp)
    print("resnet:",out.shape)
if __name__ == '__main__':
    main()
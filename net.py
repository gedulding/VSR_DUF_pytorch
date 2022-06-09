import torch
import torch.nn as nn

T_in = 7
#DUF
class DUFNET_16L(nn.Module):
    def __init__(self , uf = 4):
        super(DUFNET_16L , self).__init__();
        self.uf = uf
        self.preprocessNet = nn.Sequential(
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1, 3, 3), stride=1, padding=0),  # 帧跨度为1，卷积核为3*3*3
        )

        self.commonNet_1_1 = nn.Sequential(
            #FN and RN 0~2
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels= 64 , out_channels= 64, kernel_size=(1, 1 , 1) , stride=1 , padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 0),
            nn.Conv3d(in_channels= 64 , out_channels= 32 , kernel_size=(3,3,3) , stride=1 , padding=0),
        )

        self.commonNet_1_2 = nn.Sequential(
            # FN and RN 0~2
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.Conv3d(in_channels=96, out_channels=96, kernel_size=(1, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 0),
            nn.Conv3d(in_channels=96, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0),
        )

        self.commonNet_1_3 = nn.Sequential(
            # FN and RN 0~2
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 0),
            nn.Conv3d(in_channels=128, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0),
        )

        self.commonNet_2_1 = nn.Sequential(
            # FN and RN 3~6
            nn.BatchNorm3d(160),
            nn.ReLU(),
            nn.Conv3d(in_channels= 160, out_channels= 160, kernel_size=(1, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(160),
            nn.ReLU(),
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            nn.Conv3d(in_channels= 160, out_channels= 32, kernel_size=(3, 3, 3), stride=1, padding=0),
        )
        self.commonNet_2_2 = nn.Sequential(
            # FN and RN 3~6
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.Conv3d(in_channels=192, out_channels=192, kernel_size=(1, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            nn.Conv3d(in_channels=192, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0),
        )
        self.commonNet_2_3 = nn.Sequential(
            # FN and RN 3~6
            nn.BatchNorm3d(224),
            nn.ReLU(),
            nn.Conv3d(in_channels=224, out_channels=224, kernel_size=(1, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(224),
            nn.ReLU(),
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            nn.Conv3d(in_channels=224, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0),
        )

        self.commonNet_3 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConstantPad3d((1, 1, 1, 1, 0, 0), 0),
            nn.Conv3d(in_channels= 256, out_channels= 256, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.ReLU()
        )

        self.RedisualNet = nn.Sequential(
            nn.Conv3d(in_channels= 256 , out_channels= 256 , kernel_size=(1, 1 ,1), stride=1, padding=0),            #根据原文作者，输出通道数不一样，所以跟Filter多出来这一步
            nn.ReLU(),
            nn.Conv3d(in_channels= 256 , out_channels= 3*self.uf*self.uf , kernel_size=(1, 1 ,1), stride=1, padding=0)
        )

        self.FilterNet = nn.Sequential(
            nn.Conv3d(in_channels= 256, out_channels= 512, kernel_size=(1, 1 ,1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(in_channels= 512, out_channels= 1*5*5*self.uf*self.uf, kernel_size=(1, 1 ,1), stride=1, padding=0),
        )

        self.FilterSoft = nn.Sequential(
            nn.Softmax(dim=2)
        )

    def forward(self , x):
        x = self.preprocessNet(x)

        t = self.commonNet_1_1(x)
        x = torch.cat((x , t), dim=1)
        t = self.commonNet_1_2(x)
        x = torch.cat((x, t), dim=1)
        t = self.commonNet_1_3(x)
        x = torch.cat((x, t), dim=1)


        t = self.commonNet_2_1(x)
        x = torch.cat((x[:,:,1:-1], t), dim=1)
        t = self.commonNet_2_2(x)
        x = torch.cat((x[:,:,1:-1], t), dim=1)
        t = self.commonNet_2_3(x)
        x = torch.cat((x[:,:,1:-1], t), dim=1)

        x = self.commonNet_3(x)
        #Residual generation network
        r = self.RedisualNet(x)
        #Filter generation network
        f = self.FilterNet(x)
        ds_f = f.shape;
        f = f.reshape(ds_f[0], 25,  16, ds_f[2], ds_f[3], ds_f[4])
        f = self.FilterSoft(f)

        # cascade
        # m = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3), stride=1, padding=0)
        # for c in range(3):
        #     Target_frame = x[:, c, T_in // 2:T_in // 2 + 1, :, :]


        return f,r;
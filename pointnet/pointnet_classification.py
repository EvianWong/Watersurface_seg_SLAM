import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class_number = 1


class T_Net_3(torch.nn.Module):
    def __init__(self):
        super(T_Net_3,self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 32, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(32, 128, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(128, 1024, kernel_size=1)
        self.fc1 = torch.nn.Linear(1024,256)
        self.fc2 = torch.nn.Linear(256,64)
        self.fc3 = torch.nn.Linear(64,9)

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.bn5 = torch.nn.BatchNorm1d(64)

    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))           # bs*3*n   -> bs*32*n
        x = F.relu(self.bn2(self.conv2(x)))           # bs*32*n  -> bs*128*n
        x = F.relu(self.bn3(self.conv3(x)))           # bs*128*n -> bs*1024*n

        x = torch.max(x, 2)[0]                        # bs*1024*n -> bs*1024
        x = x.view(batch_size,-1)
        x = F.relu(self.bn4(self.fc1(x)))             # bs*1028 -> bs*256
        x = F.relu(self.bn4(self.fc2(x)))             # bs*256  -> bs*64
        x = self.fc3(x)                               # bs*64   -> bs*9

        diag = Variable(torch.from_numpy(np.eye(3, np.float32))).view(1, 9).repeat(batch_size, 1)
        x += diag
        x = x.view(-1, 3, 3)                          # bs*9 -> bs*3*3
        return x


class T_Net_64(torch.nn.Module):
    def __init__(self):
        super(T_Net_64, self).__init__()
        self.conv1 = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(256, 1024, kernel_size=1)
        self.fc1 = torch.nn.Linear(1024,512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc3 = torch.nn.Linear(256,4096)

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.bn5 = torch.nn.BatchNorm1d(64)

    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))           # bs*64*n  -> bs*128*n
        x = F.relu(self.bn2(self.conv2(x)))           # bs*128*n -> bs*256*n
        x = F.relu(self.bn3(self.conv3(x)))           # bs*256*n -> bs*1024*n

        x = torch.max(x, 2)[0]                        # bs*1024*n -> bs*1024
        x = x.view(batch_size,-1)
        x = F.relu(self.bn4(self.fc1(x)))             # bs*1028 -> bs*256
        x = F.relu(self.bn4(self.fc2(x)))             # bs*256  -> bs*256
        x = self.fc3(x)                               # bs*256  -> bs*4096

        diag = Variable(torch.from_numpy(np.eye(64, dtype=np.float32))).view(1, 4096).repeat(batch_size, 1)
        x += diag
        x = x.view(-1, 64, 64)                        # bs*4096 -> bs*64*64
        return x


class PointNet_Classification(torch.nn.Module):
    def __init__(self) -> None:
        super(PointNet_Classification, self).__init__()
        self.Tnet3 = T_Net_3()
        self.Tnet64 = T_Net_64()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        self.fc1 = torch.nn.Linear(1024,512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc3 = torch.nn.Linear(256,class_number)

    def forward(self, x):
        batch_size, _, _ = x.size()          #batchsize, channel(xyz,xyz+rgb), 点数
        trans_3d = self.Tnet3(x)
        x = torch.bmm(x, trans_3d)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_64d = self.Tnet3(x)
        x = torch.bmm(x, trans_64d)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = x.view(batch_size, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x


class PointNet_Segmentation(torch.nn.Module):
    def __init__(self) -> None:
        super(PointNet_Segmentation, self).__init__()
        self.Tnet3 = T_Net_3()
        self.Tnet64 = T_Net_64()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1088)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        self.fc1 = torch.nn.Linear(1088,512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc3 = torch.nn.Linear(256,class_number)

    def forward(self, x):
        batch_size, _, n = x.size()
        trans_3d = self.Tnet3(x)
        x = torch.bmm(x, trans_3d)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_64d = self.Tnet64(x)
        x = torch.bmm(x, trans_64d)
        point_feature = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        global_feature = torch.max(x, 2)[0]
        global_feature_1 = global_feature.view(batch_size, 1024, 1).repeat(1, 1, n)
        x = torch.cat([global_feature_1, point_feature], 1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x

def Regulization_transform(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]

    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss

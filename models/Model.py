import timm
import torch.nn as nn
import torch

class direction_model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('resnet101', pretrained=pretrained)
        #self.model.global_pool=nn.Identity()
        self.model.fc = nn.Identity()

    def forward(self, x):
        x=self.model(x)

        return x
    
class Model(nn.Module):
    def __init__(self, in_features, classes=2):
        super().__init__()
        self.fc1 = nn.Linear(20, classes, bias=True)
        self.fc2 = nn.Linear(in_features, in_features //4, bias=True)
        self.fc3 = nn.Linear(in_features //4, in_features // 8, bias=True)
        self.fc4 = nn.Linear(in_features //8, classes, bias=True)

        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.direction = direction_model()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.direction(x)
        
        f=[]

        for i in range(0, x.shape[0], 20):
            f.append(x[i:i+20])

        x = torch.stack(f,0)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)        

        x = self.fc4(x)

        x = self.softmax(x)

        return x
    

if __name__=='__main__':
    sample = torch.randn((80, 3, 256, 256))
    model = Model(2048)
    pred = model(sample)
    print(pred.shape)
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_classes):
        super(MLP, self).__init__()

        feature_dim = 27
        embed_dim = 64
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, embed_dim*2), 
            nn.LeakyReLU(0.01), 
            nn.Linear(embed_dim*2, embed_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(embed_dim, n_classes)
            )


    def forward(self, x_feature):
        
        classes = nn.functional.softmax(self.feature_mlp(x_feature), dim=1)
        
        return classes

if __name__ == '__main__':
    batch_size = 32
    num_classes = 2
    model = MLP(num_classes)    
    x_feature = torch.rand(batch_size, 27)  # 输入序列
    output = model(x_feature)
    print(output.shape)
    paras = sum([p.data.nelement() for p in model.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))



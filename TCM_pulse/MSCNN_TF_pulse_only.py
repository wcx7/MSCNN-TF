import torch
from torch import nn
from transformer_block import TransformerBlock

class MSCNNTran(nn.Module):
    def __init__(self, n_chans, n_classes):
        super(MSCNNTran, self).__init__()

        self.temp_conv1 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv2 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv3 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv4 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv5 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.temp_conv6 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2 ,groups=n_chans)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(dim=64, num_heads=4, mlp_ratio=2, dropout=0.1) for _ in range(2)])
        embed_dim = 64
        self.pos_embed = nn.Parameter(torch.zeros(1, 6, embed_dim), requires_grad=True)  # 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.chpool1    = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01))

        self.chpool2    = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01))

        self.chpool3    = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01))
            
        self.chpool4    = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01))

        self.chpool5    = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, kernel_size=4,groups=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01))

        self.classifier = nn.Linear(embed_dim, n_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(320,64),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(64,64),
        #     nn.Sigmoid(),
        #     nn.Linear(64,n_classes))


    def forward(self, x , return_attention=False):

        temp_x  = self.temp_conv1(x)               
        temp_w1 = self.temp_conv2(temp_x)         
        temp_w2 = self.temp_conv3(temp_w1)      
        temp_w3 = self.temp_conv4(temp_w2)       
        temp_w4 = self.temp_conv5(temp_w3)      
        temp_w5 = self.temp_conv6(temp_w4)  
                      

        w1 = self.chpool1(temp_w1).mean(dim=(-1))
        w2 = self.chpool2(temp_w2).mean(dim=(-1))
        w3 = self.chpool3(temp_w3).mean(dim=(-1))
        w4 = self.chpool4(temp_w4).mean(dim=(-1))
        w5 = self.chpool5(temp_w5).mean(dim=(-1))

        # 用transformer做不同尺度融合
        w = torch.stack([w1,w2,w3,w4,w5], dim=0)
        w = torch.transpose(w, 0, 1)
        w = w + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(w.shape[0], -1, -1)
        w = torch.cat((cls_tokens, w), dim=1)        
        for index_real, block in enumerate(self.transformer_blocks):
            if index_real == len(self.transformer_blocks) - 1:
                w, attn_weights = block(w)
        cls_token = w[:,0]
        # concat_vector  = torch.cat([w1,w2,w3,w4,w5],1)

        # 取cls token分类
        classes = nn.functional.softmax(self.classifier(cls_token),dim=1)
        # classes = nn.functional.softmax(self.classifier(concat_vector),dim=1)  

        if return_attention:
            return classes, attn_weights
        else:
            return classes

if __name__ == '__main__':
    batch_size = 16
    pool = 'mean'
    num_classes = 2
    model = MSCNNTran(4,num_classes)    
    input = torch.rand(batch_size, 4, 5000)  # 输入序列
    output = model(input)
    print(output.shape)
    paras = sum([p.data.nelement() for p in model.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))
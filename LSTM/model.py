import torch.nn as  nn
from transformers import BertModel, BertTokenizer, BertConfig


class testModel(nn.Module):
    def __init__(self):
        super(self).__init__()
        #BERT 层
        self.bert=BertModel.from_pretrained('/Users/jasondennis/Desktop/chinese_L-12_H-768_A-12')
        #LSTM 层
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=2,bidirectional=True)
        #Dropout 层
        self.dropout = nn.Dropout(0.3)

        self.linear1 = nn.Linear(256*2,256)
        self.linear2 =nn.Linear(256,128)
        self.linear3=nn.Linear(128,64)
        self.sigmoid = nn.Sigmoid()



    def forward(self,x，hidden):
    batch_size = x.size(0)
    #生成bert字向量
    x=self.bert(x)[0]     #bert 字向量

    # lstm_out
    #x = x.float()
    lstm_out, (hidden_last,cn_last) = self.lstm(x, hidden)
    #print(lstm_out.shape)   #[32,100,768]
    #print(hidden_last.shape)   #[4, 32, 384]
    #print(cn_last.shape)    #[4, 32, 384]

    #修改 双向的需要单独处理
    if self.bidirectional:
        #正向最后一层，最后一个时刻
        hidden_last_L=hidden_last[-2]
        #print(hidden_last_L.shape)  #[32, 384]
        #反向最后一层，最后一个时刻
        hidden_last_R=hidden_last[-1]
        #print(hidden_last_R.shape)   #[32, 384]
        #进行拼接
        hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
        #print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
    else:
        hidden_last_out=hidden_last[-1]   #[32, 384]


    # dropout and fully-connected layer
    out = self.dropout(hidden_last_out)
    #print(out.shape)    #[32,768]
    out = self.fc(out)

    return out


import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model_done import bert_lstm
from sklearn.model_selection import train_test_split


output_size = 1
hidden_dim = 384   #768/2
n_layers = 2
bidirectional = True  #这里为True，为双向LSTM

net = bert_lstm(hidden_dim, output_size,n_layers, bidirectional)

#print(net)


np.random.seed(2020)
torch.manual_seed(2020)

data=pd.read_csv('/Users/jasondennis/Desktop/train_data.csv',encoding='utf-8')

def pretreatment(comments):
    result_comments=[]
    punctuation='。，？！：%&~（）、；“”&|,.?!:%&~();""'
    for comment in comments:
        comment= ''.join([c for c in comment if c not in punctuation])
        comment= ''.join(comment.split())   #\xa0
        result_comments.append(comment)

    return result_comments

result_comments=pretreatment(list(data['text_a'].values))

len(result_comments)

result_comments[:1]
from transformers import BertTokenizer,BertModel
tokenizer = BertTokenizer.from_pretrained('/Users/jasondennis/Desktop/chinese_L-12_H-768_A-12')
result_comments_id=tokenizer(result_comments,padding=True,truncation=True,max_length=200,return_tensors='pt')

result_comments_id

X=result_comments_id['input_ids']
y=torch.from_numpy(data['label'].values).float()

X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.3,shuffle=True,stratify=y,random_state=2020)


len(X_train),len(X_test)
X_train.shape

X_valid,X_test,y_valid,y_test=train_test_split(X_test,y_test,test_size=0.5,shuffle=True,stratify=y_test,random_state=2020)

train_data = TensorDataset(X_train, y_train)
valid_data = TensorDataset(X_valid, y_valid)
test_data = TensorDataset(X_test,y_test)

# dataloaders
batch_size = 64

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,drop_last=True)


# loss and optimization functions
lr=2e-5
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params
epochs = 10
# batch_size=50
print_every = 7
clip=5 # gradient clipping

# move model to GPU, if available
if(USE_CUDA):
    net.cuda()


net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)
    counter = 0

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(USE_CUDA):
            inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.data for each in h])
        net.zero_grad()
        output= net(inputs, h)
        print(output.squeeze())
        print(labels.float())
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            net.eval()
            with torch.no_grad():
                val_h = net.init_hidden(batch_size)
                val_losses = []
                for inputs, labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])

                    if(USE_CUDA):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
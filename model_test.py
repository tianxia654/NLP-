import torch
from vocab import Vocab
from torch import nn
import numpy as np
from gensim.models import KeyedVectors
import torch.nn.functional as F
#模型搭建

def read_pretrained_wordvec(path,vocab:Vocab,word_dim):
    '''
    给vocab中的每个词分配词向量，如果有预先传入的训练好的词向量，则提取出来
    path:词向量存储路径
    vocab:词典
    word_dim:词向量的维度
    vectoPath:词向量表路径
    返回值是词典（按照序号）对应的词向量
    '''
    vecs=np.random.normal(0.0,0.9,[len(vocab),word_dim]) #先随机给词典中的每个词分一个随机向量
    # with open(path,'r',encoding='utf-8') as file:
    #     for line in file:
    #         line=line.split()
    #         if line[0] in vocab.vocab: #在词典里则提取出来，存到序号对应的那一行去
    #             vecs[vocab.word2seq(line[0])]=np.asarray(line[1:],dtype='float32')     #将vocab中的索引值换成词向量

    #vecs 中的向量顺序与vocab中单词顺序保持一致

    keyVec = KeyedVectors.load('ctest/Tencent_AILab_englishEmbedding.bin')
    for i, text in enumerate(vocab.vocab):
        if text.lower() in keyVec:
            vecs[vocab.word2seq(text)] = np.array(keyVec[text.lower()], dtype='float32')
    return vecs


class TextCNN1(nn.Module):
    def __init__(self,vecs,embedding_dimension,label_num):
        super(TextCNN1, self).__init__()

        chanel_num = 1      #输入
        filter_num = 100    #输出
        filter_sizes = [3, 4, 5]

        # 随机生成
        self.embedding=nn.Embedding(1000,100)  #有了词典大小个词向量
        self.embedding.weight.requires_grad=True#动态更改

        # self.embedding=nn.Embedding.from_pretrained(torch.from_numpy(vecs).float())

        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, label_num)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x,batch_label=None):

        x = self.embedding(x)
        x = x.unsqueeze(1)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.sigmoid(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)

        if batch_label is not None:
            loss = self.loss_fn(logits, batch_label.float())
            return loss
        else:
            # return torch.argmax(pre, dim=-1)
            return torch.sigmoid(logits)
        # output = torch.sigmoid(logits)
        # return output


class Block(nn.Module):
    def __init__(self, out_channel, max_lens, kernel_s, embed_num):
        super(Block, self).__init__()
        # 这里out_channel是卷积核的个数
        self.cnn = nn.Conv2d(1, out_channel, kernel_size=(kernel_s, embed_num))
        # self.act = nn.ReLU()
        self.maxp = nn.MaxPool1d(kernel_size=(max_lens - kernel_s + 1))

    def forward(self, emb):
        # emb.shape = torch.Size([1, 7, 5]),我们需要加一个维度1，来达到输入通道要求
        output = self.cnn(emb)
        # output.shape = torch.Size([1, 2, 6, 1])
        # output1 = self.act(output)
        output1 = F.relu(output)
        # 最大池化我们2-3个维度，所以，最后的1需要去掉
        output1 = output1.squeeze(3)
        output2 = self.maxp(output1)
        return output2.squeeze(dim=-1)




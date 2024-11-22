from torch.utils.data import Dataset
import torch
import nltk  #英文分词
import json
from tqdm import tqdm

def read_test_data(path):
    # 打开并读取JSON文件
    labels=[]
    inputs=[]
    labels_num = {}
    for i, line in enumerate(open('ctest/label_list.txt', 'r')):
        labels_num[line.replace('\n','')] = i
    if path.split(".")[1] == 'json':
        for i,line in enumerate(open(path,'r')):
            dict=json.loads(line)
            inputs.append(dict['text'])
            labels.append(dict['label'])
            # labels.append(labels_num[dict['label'][0]])      #写成标签对应的索引值(单标签)

    #多标签分类
    for i, x in enumerate(labels):
        list = [0] * len(labels_num)
        for j in range(len(x)):
            list[labels_num[x[j]]] = 1
        labels[i] = list

    return inputs,labels,len(labels_num)


#数据预处理，处理句子
class CtestDataset(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.inputs, self.labels, self.label_num=read_test_data(path)
        self.data2token()

   #原始文本转Token
    def data2token(self):
        self.avg_len=0
        x = 0
        for i, data in enumerate(tqdm(self.inputs)):  #读inputs句子，tqdm进度条看读了多少句子
            #第一次运行需要
            #nltk.download('punkt')
            self.inputs[i] = nltk.word_tokenize(data)  #英文分词
            self.avg_len+=len(self.inputs[i]) #统计句子平均长度,每个句子分了多少词
            x += 1
        self.avg_len /= x
        print(f'the average len is {self.avg_len}')

    #将词转换成seq序列，
    def token2seq(self, vocab, padding_len): #padding_len设置一个固定的输入长度，过长的句子阶段，短句用pad补充
        for i, data in enumerate(self.inputs):
            #每个句子小于padding大小，则用vocab中的<pad>填充，因为<pad>索引为0没有影响
            if len(self.inputs[i])<padding_len:
                self.inputs[i] += [vocab.padding_word]*(padding_len-len(self.inputs[i])) #填充，填充多少个东西
            elif len(self.inputs[i])>padding_len:
                #大于padding长度则直接截取到padding
                self.inputs[i]=self.inputs[i][:padding_len]

            #将每个单词转换成对应的索引
            for j in range(padding_len):
                self.inputs[i][j]=vocab.word2seq(self.inputs[i][j])
            #将每个句子设置成tensor类型
            self.inputs[i]=torch.tensor(self.inputs[i], dtype=torch.long)

            self.labels[i]=torch.tensor(self.labels[i], dtype=torch.long)

    #看有多少个样本=看有多少个标签
    def __len__(self):
        return len(self.labels)

    #一条一条读数据
    def __getitem__(self, item:int):
        return self.inputs[item], self.labels[item]


if __name__ == '__main__':
    from vocab import Vocab
    train_inputset = CtestDataset('ctest/train.json')
    vocab=Vocab(train_inputset.inputs, 10)
    print(vocab.word2seq('Changes'))



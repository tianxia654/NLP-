#构造词典进行数据预处理
#首先构造一个词典类
class Vocab():
    def __init__(self,datas:list,limit_size)->None:   #datas:原始数据list类型，limit_size:设置词典大小，不能将所有词放入，可以将词频高的词放入
        self.vocab={}#字典类
        self.padding_word='<pad>' #未知长度的一句话，进行截断或填充成为固定东西的输入
        cnt={} #临时字典，统计Token出现的频率

        #统计所有词词频
        for data in datas: #每一句话循环
            for word in data: #话里的每一个词，统计次数
                if word not in cnt:cnt[word]=1
                else:cnt[word]+=1

        self.vocab[self.padding_word]=0 #将填充的东西放在下标为0的位置

        if len(cnt)>limit_size: #如果不重复单词总数>limit_size  500
            cnt=sorted(cnt.items(), key=lambda t:t[1],reverse=True) #按照词频从大到小排序

            #建立单词的索引，一个单词对应一个数，每个单词对应出现次数的先后顺序的大小
            for w, _ in cnt:
                if len(self.vocab)==limit_size:break
                self.vocab[w]=len(self.vocab) #不再统计词频，依照下标赋值  添加一个字典大小就+1
        else:
            for w, _ in cnt.items():
                self.vocab[w] = len(self.vocab) #直接将预定义词典放到真实词典，按下标进行赋值


    #统计词典大小
    def __len__(self):
        return len(self.vocab)
    #根据原始Token找词向量，找词向量不是按照原来文本表述去找，而是根据一个数值去找，所以要把原始的词序列变成一个seq序列（数值序列）
    def word2seq(self,word)->int:
        if word not in self.vocab: #如果不在，取0，在，取所在位置
            return self.vocab[self.padding_word]
        return self.vocab[word]





from tqdm import tqdm
import nltk
from  vocab import Vocab
import torch

def runModel():
    text_path = "ctest/test.txt"
    label_path = 'ctest/label_list.txt'
    inputs = []
    text = []
    label_mapping = {}
    with open(text_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file.readlines()):
            inputs.append(line.replace('\n', ''))
            text.append(line.strip('\n'))

    for i, data in enumerate(tqdm(inputs)):  # 读inputs句子，tqdm进度条看读了多少句子
        inputs[i] = nltk.word_tokenize(data)  # 英文分词

    with open(label_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            label_mapping[i] = line.strip('\n')

    padding_len = 60

    # 1、建立单词索引，第一个为<pad> 为0
    vocab = Vocab(inputs, 1000)

    # 2.1转化成序列,判断每个句子的大小，大于padding则用vocab中的<pad>填充
    for i, data in enumerate(inputs):
        if len(inputs[i]) < padding_len:
            inputs[i] += [vocab.padding_word] * (padding_len - len(inputs[i]))  # 填充，填充多少个东西
        elif len(text[i]) > padding_len:
            inputs[i] = inputs[i][:padding_len]

        # 2.2将对应的单词转换成对应的索引
        for j in range(padding_len):
            inputs[i][j] = vocab.word2seq(inputs[i][j])
        inputs[i] = torch.tensor(inputs[i], dtype=torch.long)
        inputs[i] = torch.unsqueeze(inputs[i], 0).to("cpu")
    model_name = 'TextCNN1_model_50.pth'
    model = torch.load("models/TextCNN1/"+model_name,map_location=torch.device('cpu'))
    model.eval()

    with torch.no_grad():
        with open("predition1.txt", "w", encoding="utf-8") as file:
            for i in range(len(inputs)):
                output = model(inputs[i])
                predictions = (output > 0.3).float()
                if torch.sum(predictions) == 0:
                    # 如果所有元素都是0，将最大的数设为1
                    max_index = torch.argmax(output)
                    predictions[0][max_index] = 1
                predicted_labels = [label_mapping[idx] for idx in range(len(label_mapping)) if predictions[0][idx] == 1]
                # file.write(f"{predicted_labels}\n")
                # 如果没有预测标签，则设置默认标签
                if not predicted_labels:
                    predicted_labels = ["No_Mentioned"]

                # 将标签拼接成一个字符串，每个标签用空格分隔，并写入同一行
                file.write(",".join(predicted_labels) + "\n")
    print("结果保存完成")

if __name__ == '__main__':
    runModel()

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from data_test import CtestDataset
from model_test import TextCNN1, read_pretrained_wordvec
from vocab import Vocab

# 超参数设置
epoch = 50            # 训练次数
batch_size = 32        # 每次训练的批量大小
padding_len = 60       # 每句话的填充长度
vocab_limit = 1000     # 词典大小
word_dim = 100         # 词向量维度
label_num = 31         # 标签数量
lr = 1e-3              # 初始学习率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CtestDataset('ctest/train.json')
val_dataset = CtestDataset('ctest/valid.json')

# 构建词汇表
vocab = Vocab(train_dataset.inputs, vocab_limit)

# 转化为序列
train_dataset.token2seq(vocab, padding_len)
val_dataset.token2seq(vocab, padding_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 模型初始化
model_name = 'TextCNN1'
model = TextCNN1(read_pretrained_wordvec('ctest/dict.txt', vocab, word_dim), word_dim, label_num)

model = model.to(device)

# 优化器与学习率调度
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # 添加L2正则化
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率调度器

# TensorBoard日志
writer = SummaryWriter("logs")

# 评估函数：计算精度、召回率等
def calculate_precision_recall(predictions, labels):
    true_positives = torch.sum(predictions * labels, dim=1).float()
    false_positives = torch.sum(predictions * (1 - labels), dim=1).float()
    false_negatives = torch.sum((1 - predictions) * labels, dim=1).float()

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    return precision, recall

def evaluate(model, dataloader, device, threshold=0.3):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_precision = 0
    total_recall = 0

    with torch.no_grad():
        for batch_idx, (batch_data, batch_labels) in enumerate(tqdm(dataloader)):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            output = model(batch_data)

            # 将输出转为二进制标签
            predictions = (output > threshold).float()

            # 计算准确率、精确度、召回率
            correct_predictions += (predictions == batch_labels).all(dim=1).int().sum().item()
            total_samples += len(batch_labels)

            precision, recall = calculate_precision_recall(predictions, batch_labels)
            total_precision += precision.mean().item()
            total_recall += recall.mean().item()

    accuracy = correct_predictions / total_samples * 100
    average_precision = total_precision / len(dataloader)
    average_recall = total_recall / len(dataloader)

    print(f"验证准确率: {accuracy:.3f}%")
    print(f"平均精确度: {average_precision:.3f}")
    print(f"平均召回率: {average_recall:.3f}")

    return accuracy, average_precision, average_recall

# 训练过程
def train_multi(epochs):
    total_train_step = 0
    best_accuracy = 0

    for e in range(epochs):
        model.train()

        # 训练
        for batch_idx, (batch_data, batch_labels) in enumerate(tqdm(train_loader)):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()  # 清空梯度
            loss = model(batch_data, batch_labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            total_train_step += 1
            writer.add_scalar("train_loss", loss.item(), total_train_step)

        print(f"epoch:{e+1}, loss={loss.item():.3f}")

        # 学习率调度
        scheduler.step()

        # 每个epoch评估模型
        accuracy, average_precision, average_recall = evaluate(model, val_loader, device, threshold=0.3)

        # 记录训练过程
        writer.add_scalar("accuracy", accuracy, e)
        writer.add_scalar("average_precision", average_precision, e)
        writer.add_scalar("average_recall", average_recall, e)

        # 早停机制：保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model, f"models/{model_name}/{model_name}_model_best.pth")
            print(f"保存最佳模型，验证准确率: {best_accuracy:.3f}%")

        print(f"验证准确率: {accuracy:.3f}%")

    # 保存最终模型
    torch.save(model, f"models/{model_name}/{model_name}_model_{epochs}.pth")
    print("训练完成，模型保存完成")
    writer.close()

if __name__ == '__main__':
    train_multi(epoch)

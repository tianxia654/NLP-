# NLP-
ctest：
* label_list.txt：标签列表  
* test.txt：预测内容  
* train.json、valid.json：训练集和验证集  

logs：tensorboard图像保存位置  
models：训练模型保存位置  
data_test.py：数据的处理  
model_test.py：模型结构  
predict.py：内容的标签预测  
train_test.py：模型的训练  
vocab.py：构建词典


 # 使用方法
1、模型的训练：train_test.py中直接调用主函数  
2、模型的使用：predict.py直接调用主函数

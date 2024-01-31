# 多模态情感分析

## 实验任务

- 给定配对的文本和图像，预测对应的情感标签。
- 三分类任务：positive, neutral, negative。

## 实验数据集

- data文件夹：包括所有的训练文本和图片，每个文件按照唯一的guid命名。
- train.txt: 数据的guid和对应的情感标签。
- test_without_label.txt：数据的guid和空的情感标签。

## 文件结构

```
|-- multi_model.py		多模态模型结构定义
|-- multi_train.py		训练多模态模型
|-- only_train.py		消融实验
|-- predict.py	        预测测试集的情感标签
|-- README.md
|-- requirements.txt    
|-- 实验报告.pdf
|-- test_predictions.txt	测试集预测结果
|-- src
	|-- 实验报告中用到的图片
```

## 参考库

tqdm==4.66.1

transformers==4.36.2

scikit-learn==1.3.1
chardet==5.2.0

Pillow==10.0.1

numpy==1.26.0

matplotlib==3.8.0

torch == 2.1.1
torchvision == 0.16.1

```
pip install -r requirements.txt
```
注：requirements.txt中torch 和 torchvision这两个库是手动下载的，可单独使用以下命令单独安装

```
pip install torch==2.1.1 torchvision==0.16.1
```

## 运行代码流程

### 1 .  数据与预训练模型下载

[下载地址](https://pan.baidu.com/s/15ULV1ltwfWVUWNp1tS9EHg?pwd=a24w)

提取码：a24w 

将数据压缩文件解压，和模型文件夹bert_base_cased一起放在项目根目录下，如下图
![Alt](/src/setting.png)

### 2 . 训练多模态模型：

```
python multi_train.py --lr 0.000003 --bert_lr 0.0000005 --resnet_lr 0.0000005 --weight_decay 0.0 --epoch 20 --batch_size 16 --attention_nhead 8 --train_percent 0.8 --hidden_dim 64 --model_class attention_cat
```

其中，bert_lr、resnet_lr为整个模型中bert和resnet模型参数训练时的学习率，lr为模型中其他参数的学习率，hidden_dim 为多模态模型中线性分类器(2层MLP)隐层大小，model_class为模型类型，可选择decision_avg 或 attention_cat .

训练好的模型会保存在项目根目录下，模型文件名为 'best_model.pt'


### 3 . 载入模型预测测试文件标签

载入训练好的模型的参数(模型类型model_class，隐层大小 hidden_dim，多头注意力中的nhead)要与训练模型时的参数一致

```
 python predict.py --attention_nhead 8 --hidden_dim 64 --model_class attention_cat
```

预测结果保存在test_predictions.txt中。
### 4 . 单模态消融实验：

仅文本 

```
python only_train.py --lr 0.000003 --bert_lr 0.0000005 --resnet_lr 0.0000005 --weight_decay 0.0 --epoch 20 --batch_size 16 --attention_nhead 8 --train_percent 0.8 --hidden_dim 64 --model_class text 
```

仅图像

```
python only_train.py --lr 0.000003 --bert_lr 0.0000005 --resnet_lr 0.0000005 --weight_decay 0.0 --epoch 20 --batch_size 16 --attention_nhead 8 --train_percent 0.8 --hidden_dim 64 --model_class image 
```



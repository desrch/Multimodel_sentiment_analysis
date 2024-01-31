import os
import chardet
import numpy as np
import torch
import random
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from torchvision.models import  resnet101
from PIL import Image
from multi_model import Multimodel_attention_cat_trans, Multimodel_decision_weighted_avg
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import argparse


def args_parse():
    parse = argparse.ArgumentParser(description='add parameter')
    parse.add_argument('--lr', default=3e-5,type=float)
    parse.add_argument('--bert_lr',default=5e-6,type=float)
    parse.add_argument('--resnet_lr',default=5e-6,type=float)
    parse.add_argument('--weight_decay',default=0.0,type=float)
    parse.add_argument('--epoch', default=20,type=int)
    parse.add_argument('--batch_size', default=16,type=int)
    parse.add_argument('--attention_nhead', default=8,type=int)
    parse.add_argument('--train_percent', default=0.80,type=float)
    parse.add_argument('--hidden_dim', default=64,type=int)
    parse.add_argument('--model_class', default='attention_cat')
    args = parse.parse_args()
    return args


def get_train_id_tag(train_file):
    train_id = []
    train_tag = {}
    with open(train_file, 'r') as f:
        f.readline()
        for line in f.readlines():
            id, tag = line.strip().split(',')
            train_id.append(id)
            train_tag[id] = int(label_dict[tag])
    return train_id, train_tag


def get_test_id(test_file):
    test_id_list = []
    with open(test_file, 'r') as f:
        f.readline()
        for line in f.readlines():
            test_id_list.append(line.strip().split(',')[0])
    return test_id_list


def collate_fn(batch):
    texts = [torch.LongTensor(b[0]) for b in batch]
    images = torch.FloatTensor([np.array(b[1]).tolist() for b in batch])
    labels = torch.LongTensor([b[2] for b in batch])

    texts_mask = [torch.ones_like(text) for text in texts]
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    padded_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)
    return padded_texts, images, labels, padded_mask


def train(train_data_loader, model, optimizer, loss_fun):
    model.train()
    total_loss = 0.0
    cnt = 0
    for batch in train_data_loader:
        cnt += 1
        batch_texts, batch_images, batch_labels, batch_mask = batch
        batch_texts = batch_texts.to(device)
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        batch_mask = batch_mask.to(device)

        optimizer.zero_grad()
        logits = model(batch_texts, batch_images, batch_mask)
        batch_labels_one_hot = F.one_hot(batch_labels, num_classes=3).float().to(torch.float64)
        batch_labels_one_hot = batch_labels_one_hot.to(device)

        loss = loss_fun(logits, batch_labels_one_hot)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    avg_loss = total_loss / cnt
    return avg_loss


def val(train_data_loader, model, optimizer, loss_fun):
    model.eval()
    total_loss = 0.0
    cnt = 0
    predictions = []
    labels = []
    for batch in train_data_loader:
        cnt += 1
        batch_texts, batch_images, batch_labels, batch_mask = batch
        batch_texts = batch_texts.to(device)
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        batch_mask = batch_mask.to(device)

        logits = model(batch_texts, batch_images, batch_mask)
        prediction = torch.argmax(logits, dim=1)

        predictions.extend(prediction.cpu().tolist())
        labels.extend(batch_labels.cpu().tolist())
        batch_labels_one_hot = F.one_hot(batch_labels, num_classes=3).float().to(torch.float64)
        batch_labels_one_hot = batch_labels_one_hot.to(device)
        loss = loss_fun(logits, batch_labels_one_hot)
        total_loss += loss.item()

    accuracy = accuracy_score(labels, predictions)
    avg_loss = total_loss / cnt
    return avg_loss, accuracy


if __name__ == '__main__':
    args = args_parse()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("训练设备:{}".format(device))
    pwd = os.getcwd()
    best_model_path = os.path.join(pwd, 'best_model.pt')
    data_path = os.path.join(pwd, 'data')
    train_file = os.path.join(pwd, 'train.txt')
    test_file = os.path.join(pwd, 'test_without_label.txt')
    pretrained_bert = os.path.join(pwd, 'bert_base_cased')
    label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)
    bert_model = AutoModel.from_pretrained(pretrained_bert)

    resnet = resnet101(pretrained=True)

    train_id, train_tag = get_train_id_tag(train_file)
    test_id = get_test_id(test_file)

    random.shuffle(train_id)
    train_percent = args.train_percent
    train_size = int(train_percent * len(train_id))
    train_dataset = train_id[:train_size]
    val_dataset = train_id[train_size:]

    train_texts = []
    train_images = []
    train_labels = []
    failed_id = []
    for id in train_dataset:
        text_path = os.path.join(data_path, f"{id}.txt")
        image_path = os.path.join(data_path, f"{id}.jpg")

        with open(text_path, 'rb') as f:
            content = f.read()
            result = chardet.detect(content)
            encoding = result['encoding']
        try:
            with open(text_path, 'r', encoding=encoding) as f:
                text = f.read()
                text.replace('#', '')
                tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
                tokenized_text = tokenizer.convert_tokens_to_ids(tokens)

        except UnicodeDecodeError as e:
            failed_id.append(id)

        image = Image.open(image_path).convert("RGB")
        image = image_transform(image)

        train_texts.append(tokenized_text)
        train_images.append(image)
        train_labels.append(train_tag[id])

    val_texts = []
    val_images = []
    val_labels = []
    for id in val_dataset:
        text_path = os.path.join(data_path, f"{id}.txt")
        image_path = os.path.join(data_path, f"{id}.jpg")

        with open(text_path, 'rb') as f:
            content = f.read()
            result = chardet.detect(content)
            encoding = result['encoding']
        try:
            with open(text_path, 'r', encoding=encoding) as f:
                text = f.read()
                text.replace('#', '')
                tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
                tokenized_text = tokenizer.convert_tokens_to_ids(tokens)

        except UnicodeDecodeError as e:
            failed_id.append(id)

        image = Image.open(image_path).convert("RGB")
        image = image_transform(image)

        val_texts.append(tokenized_text)
        val_images.append(image)
        val_labels.append(train_tag[id])

    train_dataset = list(zip(train_texts, train_images, train_labels))
    val_dataset = list(zip(val_texts, val_images, val_labels))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                    collate_fn=collate_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                  collate_fn=collate_fn)

    if args.model_class == 'decision_avg':
        model = Multimodel_decision_weighted_avg(bert_model, resnet, num_labels=3, hidden_dim=args.hidden_dim, )
    elif args.model_class == 'attention_cat':
        model = Multimodel_attention_cat_trans(bert_model, resnet, num_labels=3,
                                               hidden_dim=args.hidden_dim, attention_nhead=args.attention_nhead)
    bert_params = set(model.bert_model.parameters())
    resnet_params = set(model.resnet_model.parameters())
    other_params = list(set(model.parameters()) - bert_params - resnet_params)
    no_decay = ['bias', 'LayerNorm.weight']
    bert_learning_rate = args.bert_lr
    resnet_learning_rate = args.bert_lr
    learning_rate = args.lr
    weight_decay = args.weight_decay

    params = [
        {'params': [p for n, p in model.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': bert_learning_rate, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': bert_learning_rate, 'weight_decay': 0.0},
        {'params': [p for n, p in model.resnet_model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'lr': resnet_learning_rate, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.resnet_model.named_parameters() if
                    any(nd in n for nd in no_decay)],
         'lr': resnet_learning_rate, 'weight_decay': 0.0},
        {'params': other_params,
         'lr': learning_rate, 'weight_decay': weight_decay},
    ]

    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    num_epochs = args.epoch
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device)

    best_accuracy = 0
    val_accuracy = []
    print(args.model_class)
    for epoch in range(num_epochs):
        print("Epoch :{}".format(epoch + 1))
        train_loss = train(train_data_loader, model, optimizer, loss_fn)
        val_loss, accuracy = val(val_data_loader, model, optimizer, loss_fn)
        val_accuracy.append(accuracy)
        if epoch == 0:
            best_accuracy = accuracy
        elif accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
        print("训练集损失: {}".format(train_loss))
        print("验证集损失: {}".format(val_loss))
        print("验证集准确率 : {}".format(accuracy))

    print("最佳模型准确率： {}".format(best_accuracy))
    plt.plot(range(1, num_epochs + 1, 1), val_accuracy, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

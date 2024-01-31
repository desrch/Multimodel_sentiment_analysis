import argparse
import os
import chardet
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from torchvision.models import resnet101
from PIL import Image
from multi_model import Multimodel_attention_cat_trans, Multimodel_decision_weighted_avg
import torchvision.transforms as transforms


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
    texts_mask = [torch.ones_like(text) for text in texts]
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    padded_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)
    return padded_texts, images, padded_mask


def args_parse():
    parse = argparse.ArgumentParser(description='add parameter')
    parse.add_argument('--attention_nhead', default=8, type=int)
    parse.add_argument('--hidden_dim', default=64, type=int)
    parse.add_argument('--model_class', default='attention_cat')
    args = parse.parse_args()
    return args


if __name__ == '__main__':

    pwd = os.getcwd()
    args = args_parse()
    pretrained_bert = os.path.join(pwd, 'bert_base_cased')
    test_file = os.path.join(pwd, 'test_without_label.txt')
    data_path = os.path.join(pwd, 'data')
    label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.483, 0.452, 0.400], std=[0.224, 0.224, 0.224])

    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)
    bert_model = AutoModel.from_pretrained(pretrained_bert)
    resnet = resnet101(pretrained=True)
    if args.model_class == 'decision_avg':
        model = Multimodel_decision_weighted_avg(bert_model, resnet, num_labels=3, hidden_dim=args.hidden_dim, )
    elif args.model_class == 'attention_cat':
        model = Multimodel_attention_cat_trans(bert_model, resnet, num_labels=3, hidden_dim=args.hidden_dim,
                                               attention_nhead=args.attention_nhead)
    model.to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=lambda storage, loc: storage.cuda(0)), )

    test_id_list = get_test_id(test_file)
    failed_id = []
    test_text = []
    test_image = []

    for id in test_id_list:
        text_file = os.path.join(data_path, f"{id}.txt")
        image_file = os.path.join(data_path, f"{id}.jpg")

        with open(text_file, 'rb') as f:
            content = f.read()
            result = chardet.detect(content)
            encoding = result['encoding']
        try:
            with open(text_file, 'r', encoding=encoding) as f:
                text = f.read()
                text.replace('#', '')
                tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
                tokenized_text = tokenizer.convert_tokens_to_ids(tokens)
        except UnicodeDecodeError as e:
            failed_id.append(id)

        image = Image.open(image_file).convert("RGB")
        image = image_transform(image)

        test_image.append(image)
        test_text.append(tokenized_text)

    test_dataset = list(zip(test_text, test_image))
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False,
                                                   collate_fn=collate_fn)
    test_predictions = []
    model.eval()

    with torch.no_grad():
        for batch in test_data_loader:
            batch_texts, batch_images, batch_mask = batch
            batch_texts = batch_texts.to(device)
            batch_images = batch_images.to(device)
            batch_mask = batch_mask.to(device)

            logits = model(batch_texts, batch_images, batch_mask)
            prediction = torch.argmax(logits, dim=1)
            test_predictions.extend(prediction.tolist())

    with open('test_predictions.txt', 'w') as f:
        f.write('guid,tag\n')
        for id, prediction in zip(test_id_list, test_predictions):
            f.write(f"{id},{label_dict[prediction]}\n")
    print("预测完毕！")